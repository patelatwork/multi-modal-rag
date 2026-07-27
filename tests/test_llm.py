"""Chat-client resilience.

Every case here mirrors a failure seen against the live provider: a retired
model, a token-per-minute rate limit, a per-request image cap, and a payload
larger than the tier allows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multimodal_rag.config import Settings
from multimodal_rag.exceptions import ConfigurationError, LLMError
from multimodal_rag.models.llm import (
    VisionChatModel,
    _is_image_limit_error,
    _is_too_large_error,
    _is_transient,
    retry_after_seconds,
)
from multimodal_rag.utils.images import load_image_data_url


class _Response:
    def __init__(self, headers: dict[str, str], status_code: int = 429) -> None:
        self.headers = headers
        self.status_code = status_code


class _Error(Exception):
    def __init__(self, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        if headers is not None:
            self.response = _Response(headers)


class TestRetryAfter:
    def test_prefers_the_header(self) -> None:
        assert retry_after_seconds(_Error("rate limited", {"retry-after": "30"})) == 30.0

    def test_falls_back_to_the_message_body(self) -> None:
        # Groq states the delay in prose when it omits the header.
        error = _Error("Rate limit reached ... Please try again in 21.9s. Upgrade...")
        assert retry_after_seconds(error) == pytest.approx(21.9)

    def test_parses_milliseconds(self) -> None:
        assert retry_after_seconds(_Error("try again in 500ms")) == pytest.approx(0.5)

    def test_returns_none_when_unknown(self) -> None:
        assert retry_after_seconds(_Error("something else")) is None

    def test_ignores_an_unparseable_header(self) -> None:
        error = _Error("try again in 12s", {"retry-after": "not-a-number"})
        assert retry_after_seconds(error) == pytest.approx(12.0)


def _groq_error(error_type, status_code: int, message: str):
    """Build a real groq exception; its constructor needs an httpx response."""
    import httpx

    response = httpx.Response(
        status_code, request=httpx.Request("POST", "https://api.groq.com/v1/chat")
    )
    return error_type(message, response=response, body=None)


class TestErrorClassification:
    def test_rate_limit_is_transient(self) -> None:
        import groq

        assert _is_transient(_groq_error(groq.RateLimitError, 429, "limited")) is True

    def test_missing_model_is_not_transient(self) -> None:
        import groq

        # A retired model name fails identically forever; retrying wastes time.
        error = _groq_error(groq.NotFoundError, 404, "no such model")
        assert _is_transient(error) is False

    def test_auth_failure_is_not_transient(self) -> None:
        import groq

        assert _is_transient(_groq_error(groq.AuthenticationError, 401, "bad key")) is False

    def test_server_error_is_transient(self) -> None:
        import groq

        assert _is_transient(_groq_error(groq.InternalServerError, 503, "unavailable")) is True

    def test_unrelated_exception_is_not_transient(self) -> None:
        assert _is_transient(ValueError("nope")) is False

    @pytest.mark.parametrize(
        "message",
        ["Too many images provided.", "This model supports up to 3 images"],
    )
    def test_detects_image_limit(self, message: str) -> None:
        assert _is_image_limit_error(_Error(message))

    @pytest.mark.parametrize(
        "message",
        [
            "Request too large for model",
            "please reduce your message size and try again",
            "maximum context length exceeded",
        ],
    )
    def test_detects_payload_too_large(self, message: str) -> None:
        assert _is_too_large_error(_Error(message))

    def test_ordinary_error_is_neither(self) -> None:
        error = _Error("invalid api key")
        assert not _is_image_limit_error(error)
        assert not _is_too_large_error(error)


class TestConfiguration:
    def test_missing_api_key_fails_fast_and_clearly(self, settings: Settings) -> None:
        settings.groq_api_key = None
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            VisionChatModel(settings)


class TestGracefulDegradation:
    """A too-large request is shrunk and retried instead of failing outright."""

    @pytest.fixture
    def model(self, settings: Settings, figure_element, monkeypatch) -> VisionChatModel:
        monkeypatch.setattr(VisionChatModel, "_build_client", lambda self: object())
        return VisionChatModel(settings)

    def test_drops_an_image_when_the_cap_is_exceeded(
        self, model: VisionChatModel, figure_element, monkeypatch
    ) -> None:
        attempts: list[int] = []

        def fake_invoke(messages):
            images = sum(
                1
                for block in messages[-1].content
                if isinstance(block, dict) and block.get("type") == "image_url"
            )
            attempts.append(images)
            if images > 1:
                raise LLMError("Too many images provided. This model supports up to 1 images")
            return "ok"

        monkeypatch.setattr(model, "_invoke", fake_invoke)
        result = model.complete("q", images=[figure_element.image_path] * 3)

        assert result == "ok"
        assert attempts == [3, 2, 1], "should shed one image per attempt"

    def test_shrinks_context_when_the_payload_is_too_large(
        self, model: VisionChatModel, monkeypatch
    ) -> None:
        lengths: list[int] = []

        def fake_invoke(messages):
            text = messages[-1].content[0]["text"]
            lengths.append(len(text))
            if len(text) > 2000:
                raise LLMError("Request too large ... please reduce your message size")
            return "ok"

        monkeypatch.setattr(model, "_invoke", fake_invoke)
        assert model.complete("x" * 8000) == "ok"
        assert lengths[0] > lengths[-1], "prompt should have been trimmed"

    def test_gives_up_when_shrinking_cannot_help(self, model: VisionChatModel, monkeypatch) -> None:
        def fake_invoke(messages):
            raise LLMError("invalid api key")

        monkeypatch.setattr(model, "_invoke", fake_invoke)
        with pytest.raises(LLMError, match="invalid api key"):
            model.complete("short prompt")


class TestImageEncoding:
    def test_downscales_a_large_image(self, tmp_path: Path) -> None:
        from PIL import Image

        path = tmp_path / "big.png"
        Image.new("RGB", (4000, 3000), "white").save(path)

        url = load_image_data_url(path, max_dimension=800)
        assert url.startswith("data:image/jpeg;base64,")
        # Downscaling is the main lever on token cost, so it must actually shrink.
        assert len(url) < len(load_image_data_url(path, max_dimension=2000))

    def test_preserves_transparency_as_png(self, tmp_path: Path) -> None:
        from PIL import Image

        path = tmp_path / "alpha.png"
        Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(path)
        assert load_image_data_url(path).startswith("data:image/png;base64,")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from multimodal_rag.exceptions import MultimodalRagError

        with pytest.raises(MultimodalRagError, match="not found"):
            load_image_data_url(tmp_path / "absent.png")
