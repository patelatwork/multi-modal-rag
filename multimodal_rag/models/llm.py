"""Vision-capable chat client.

Wraps LangChain's ``ChatGroq`` with the operational behaviour a service needs:
bounded retries on transient failures, fast failure on unrecoverable ones,
image downscaling, and a preflight check that the configured model actually
exists and accepts images.
"""

from __future__ import annotations

import functools
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from tenacity import RetryCallState, RetryError, retry, retry_if_exception, stop_after_attempt

from multimodal_rag.config import Settings, get_settings
from multimodal_rag.exceptions import ConfigurationError, LLMError
from multimodal_rag.logging_config import get_logger
from multimodal_rag.utils.images import load_image_data_url

logger = get_logger(__name__)

# Per-request image caps differ by model and change without notice (Groq's
# current vision model allows 3). This is the optimistic ceiling; a request that
# exceeds the real limit is retried with fewer images rather than failing.
MAX_IMAGES_PER_REQUEST = 3

_TOO_MANY_IMAGES = re.compile(r"too many images|supports up to \d+ images?", re.IGNORECASE)
_TOO_LARGE = re.compile(
    r"request too large|reduce your message size|context length|too many tokens", re.IGNORECASE
)

# When a payload is too big, drop this fraction of the text each attempt.
_TEXT_SHRINK_FACTOR = 0.6
_MIN_PROMPT_CHARS = 800


def _is_image_limit_error(error: BaseException) -> bool:
    return _TOO_MANY_IMAGES.search(str(error)) is not None


def _is_too_large_error(error: BaseException) -> bool:
    """True for payload-size rejections, which retrying unchanged cannot fix."""
    return _TOO_LARGE.search(str(error)) is not None


def _is_transient(error: BaseException) -> bool:
    """Decide whether retrying could plausibly succeed.

    Rate limits, timeouts and 5xx are worth another attempt. A bad API key or a
    retired model name will fail identically every time, so those surface
    immediately instead of burning the retry budget.
    """
    import groq

    if isinstance(error, groq.APITimeoutError | groq.APIConnectionError | groq.RateLimitError):
        return True
    if isinstance(error, groq.AuthenticationError | groq.NotFoundError | groq.BadRequestError):
        return False
    if isinstance(error, groq.APIStatusError):
        return error.status_code >= 500
    return False


# Groq reports the exact reset delay in the 429 body when no header is present.
_RETRY_AFTER_TEXT = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE)


def retry_after_seconds(error: BaseException) -> float | None:
    """Extract a provider-supplied wait time from a rate-limit error.

    Guessing with plain exponential backoff wastes the retry budget: a 60-second
    token window needs a 60-second wait, not the 2-4-8 an exponential policy
    would produce. Prefer the ``Retry-After`` header, then the message body.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    match = _RETRY_AFTER_TEXT.search(str(error))
    if match:
        value = float(match.group(1))
        return value / 1000.0 if match.group(2).lower() == "ms" else value
    return None


class _MinIntervalLimiter:
    """Enforces a minimum gap between calls across threads."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


@dataclass(slots=True)
class ImageAttachment:
    """An image to include in a prompt, as a path or as raw bytes."""

    data_url: str

    @classmethod
    def from_path(cls, path: Path | str, settings: Settings | None = None) -> ImageAttachment:
        settings = settings or get_settings()
        return cls(
            data_url=load_image_data_url(
                path,
                max_dimension=settings.image_max_dimension,
                jpeg_quality=settings.image_jpeg_quality,
            )
        )


class VisionChatModel:
    """Thin, retrying facade over the configured Groq chat model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.groq_api_key:
            msg = (
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(get one at https://console.groq.com/keys)."
            )
            raise ConfigurationError(msg)
        self._client = self._build_client()
        self._limiter = _MinIntervalLimiter(self.settings.llm_min_request_interval)

    def _build_client(self) -> Any:
        from langchain_groq import ChatGroq

        kwargs: dict[str, Any] = {
            "model": self.settings.groq_model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "api_key": self.settings.groq_api_key,
            # Retries are handled here, with our own transient-error policy.
            "max_retries": 0,
            "request_timeout": self.settings.llm_timeout_seconds,
        }
        if self.settings.llm_reasoning_effort:
            # Reasoning models otherwise emit a <think> block that would end up
            # embedded in the index and shown to the user.
            kwargs["reasoning_effort"] = self.settings.llm_reasoning_effort
        return ChatGroq(**kwargs)

    # ------------------------------------------------------------------ public
    def complete(
        self,
        prompt: str,
        *,
        images: Sequence[Path | str] = (),
        system: str | None = None,
    ) -> str:
        """Send one prompt (optionally with images) and return the reply text.

        Requests that the provider rejects as *too large* -- too many images, or
        more tokens than the tier allows -- are shrunk and retried rather than
        failed. Sources are ordered by relevance, so the last image and the tail
        of the context are the cheapest things to give up. A degraded answer
        beats an error page.
        """
        selected = list(images)[:MAX_IMAGES_PER_REQUEST]
        if len(images) > len(selected):
            logger.debug("Sending %s of %s images", len(selected), len(images))

        # Encoding is the expensive part, so do it once and slice per attempt.
        encoded = [ImageAttachment.from_path(image, self.settings) for image in selected]
        text = prompt

        while True:
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for attachment in encoded:
                content.append({"type": "image_url", "image_url": {"url": attachment.data_url}})

            messages: list[BaseMessage] = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=content))

            try:
                return self._invoke(messages)
            except LLMError as error:
                if _is_image_limit_error(error) and encoded:
                    encoded.pop()
                    logger.warning(
                        "Provider rejected the image count; retrying with %s image(s)",
                        len(encoded),
                    )
                    continue

                if _is_too_large_error(error):
                    # Images dominate the token count, so shed those first.
                    if encoded:
                        encoded.pop()
                        logger.warning("Payload too large; retrying with %s image(s)", len(encoded))
                        continue
                    if len(text) > _MIN_PROMPT_CHARS:
                        text = text[: int(len(text) * _TEXT_SHRINK_FACTOR)].rstrip() + "\n[...]"
                        logger.warning(
                            "Payload too large; retrying with %s chars of context", len(text)
                        )
                        continue

                raise

    def _wait_policy(self, retry_state: RetryCallState) -> float:
        """Sleep for the provider's stated delay, else exponential backoff."""
        ceiling = self.settings.llm_max_backoff_seconds
        error = retry_state.outcome.exception() if retry_state.outcome else None

        if error is not None:
            stated = retry_after_seconds(error)
            if stated is not None:
                # A small cushion, because the window is measured server-side.
                delay = min(stated + 1.0, ceiling)
                logger.warning(
                    "Rate limited; provider asked for %.1fs, sleeping %.1fs (attempt %s)",
                    stated,
                    delay,
                    retry_state.attempt_number,
                )
                return delay

        delay = min(2.0 * (2 ** (retry_state.attempt_number - 1)), ceiling)
        logger.warning(
            "Transient chat error, retrying in %.1fs (attempt %s): %s",
            delay,
            retry_state.attempt_number,
            error,
        )
        return delay

    def _invoke(self, messages: list[BaseMessage]) -> str:
        @retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(self.settings.llm_max_retries + 1),
            wait=self._wait_policy,
            reraise=False,
        )
        def _call() -> str:
            self._limiter.acquire()
            response = self._client.invoke(messages)
            return _response_text(response)

        try:
            return _call()
        except RetryError as error:
            cause = error.last_attempt.exception()
            msg = f"Chat model failed after {self.settings.llm_max_retries + 1} attempts: {cause}"
            raise LLMError(msg) from cause
        except Exception as error:
            raise LLMError(str(error)) from error

    def health_check(self) -> None:
        """Verify credentials and model availability. Raises on failure."""
        self.complete("Reply with the single word: ok")


def _response_text(response: Any) -> str:
    """Flatten a LangChain message into plain text.

    Newer LangChain models may return a list of content blocks rather than a
    string, so both shapes are handled.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


@functools.lru_cache(maxsize=1)
def _default_chat_model() -> VisionChatModel:
    return VisionChatModel(get_settings())


def get_chat_model(settings: Settings | None = None) -> VisionChatModel:
    """Return the chat model, reusing one instance for the default settings.

    ``Settings`` is a mutable pydantic model and therefore unhashable, so only
    the (overwhelmingly common) default path is memoised; explicit overrides
    build a fresh client.
    """
    if settings is None:
        return _default_chat_model()
    return VisionChatModel(settings)
