"""Shared fixtures.

Every fixture points the application at a temporary directory, so the test
suite never touches the developer's real index, and no test needs network
access or an API key.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from multimodal_rag.config import Settings
from multimodal_rag.domain import Element, ElementKind, IndexedElement
from .fixtures.sample_pdf import build_sample_pdf


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A generated PDF with two-column prose, a chart and a rule-less table."""
    directory = tmp_path_factory.mktemp("pdfs")
    return build_sample_pdf(directory / "sample_report.pdf")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings with all storage under ``tmp_path``."""
    config = Settings(
        input_pdf_dir=tmp_path / "pdfs",
        extraction_dir=tmp_path / "extracted",
        vector_store_dir=tmp_path / "chroma",
        docstore_dir=tmp_path / "docstore",
        groq_api_key="test-key-not-used",
        collection_name="test_collection",
        summarize_text_chunks=False,
    )
    config.ensure_directories()
    return config


@pytest.fixture
def text_element() -> Element:
    return Element(
        element_id="doc1:text:0",
        document_id="doc1",
        kind=ElementKind.TEXT,
        text="Particulate concentrations peaked during the winter months.",
        page_number=1,
    )


@pytest.fixture
def figure_element(settings: Settings) -> Element:
    from PIL import Image

    # Written under the configured image root, matching where the extractor
    # puts real crops -- the API refuses to serve anything outside it.
    image = settings.images_dir / "doc1" / "figure.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), "white").save(image)
    return Element(
        element_id="doc1:figure:0",
        document_id="doc1",
        kind=ElementKind.FIGURE,
        text="",
        page_number=2,
        image_path=image,
        caption="Fig. 1. Monthly PM2.5 averages.",
    )


@pytest.fixture
def indexed_items(text_element: Element, figure_element: Element) -> list[IndexedElement]:
    return [
        IndexedElement(element=text_element, summary=text_element.text),
        IndexedElement(
            element=figure_element,
            summary="A line chart of monthly PM2.5 averages peaking in winter.",
        ),
    ]


class FakeChatModel:
    """Stand-in for :class:`VisionChatModel` that records what it was asked."""

    def __init__(self, reply: str = "stubbed summary") -> None:
        self.reply = reply
        self.calls: list[tuple[str, list]] = []

    def complete(self, prompt: str, *, images=(), system=None) -> str:
        self.calls.append((prompt, list(images)))
        return self.reply


@pytest.fixture
def fake_model() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture
def stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace the transformer with a deterministic hash-based embedding.

    Keeps the storage tests fast and offline; they exercise persistence and
    filtering, not embedding quality.
    """
    import hashlib

    from langchain_core.embeddings import Embeddings

    class HashEmbeddings(Embeddings):
        dimension = 32

        def _embed(self, text: str) -> list[float]:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = [digest[i % len(digest)] / 255.0 for i in range(self.dimension)]
            norm = sum(value * value for value in values) ** 0.5 or 1.0
            return [value / norm for value in values]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._embed(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._embed(text)

    monkeypatch.setattr(
        "multimodal_rag.storage.vector_store.get_embeddings", lambda _=None: HashEmbeddings()
    )
    yield
