"""Core domain models shared by every layer.

These are deliberately transport-agnostic: the extractors produce them, the
pipelines consume them, and the API maps them onto response schemas. Nothing
here imports LangChain, FastAPI, or a PDF library.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Self


class ElementKind(StrEnum):
    """The three modalities the system indexes."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"

    @property
    def is_visual(self) -> bool:
        """True when the element has a rendered image the vision model can read."""
        return self in (ElementKind.TABLE, ElementKind.FIGURE)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A rectangle in PDF user-space points, origin top-left."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def expanded(self, pad: float) -> BoundingBox:
        return BoundingBox(self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad)

    def merged(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def intersects(self, other: BoundingBox, tolerance: float = 0.0) -> bool:
        return not (
            self.x1 + tolerance < other.x0
            or other.x1 + tolerance < self.x0
            or self.y1 + tolerance < other.y0
            or other.y1 + tolerance < self.y0
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(slots=True)
class Element:
    """One extracted unit of a PDF: a text chunk, a table, or a figure.

    ``text`` is whatever the parser could read directly (may be empty for a
    figure). ``image_path`` points at a rendered PNG crop and is what makes an
    element visible to the vision model.
    """

    element_id: str
    document_id: str
    kind: ElementKind
    text: str = ""
    page_number: int | None = None
    image_path: Path | None = None
    caption: str | None = None
    bbox: BoundingBox | None = None
    source_pdf: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)

    @property
    def has_image(self) -> bool:
        return self.image_path is not None and self.image_path.is_file()

    def retrieval_text(self, summary: str | None = None) -> str:
        """The text that actually gets embedded.

        A caption is prepended because figure captions carry most of the
        keyword signal ("Fig. 3. RMSE by time horizon") that a user query
        will match against.
        """
        parts: list[str] = []
        if self.page_number is not None:
            parts.append(f"[page {self.page_number}]")
        if self.caption:
            parts.append(self.caption)
        body = (summary or "").strip() or self.text.strip()
        if body:
            parts.append(body)
        return "\n".join(parts).strip()


@dataclass(slots=True)
class IndexedElement:
    """An :class:`Element` paired with the summary that was embedded for it."""

    element: Element
    summary: str

    @property
    def element_id(self) -> str:
        return self.element.element_id


@dataclass(slots=True)
class RetrievedSource:
    """A retrieval hit, enriched for display in the chat UI."""

    element_id: str
    document_id: str
    kind: ElementKind
    text: str
    page_number: int | None = None
    caption: str | None = None
    image_path: Path | None = None
    score: float | None = None
    source_pdf: str | None = None

    @property
    def has_image(self) -> bool:
        return self.image_path is not None and Path(self.image_path).is_file()


@dataclass(slots=True)
class Answer:
    """The synthesised answer plus the evidence it was grounded in."""

    question: str
    text: str
    sources: list[RetrievedSource] = field(default_factory=list)
    images_sent_to_model: int = 0
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class IngestionReport:
    """Summary of one ingestion run, surfaced by both the CLI and the API."""

    document_id: str
    filename: str
    pages: int
    text_chunks: int = 0
    tables: int = 0
    figures: int = 0
    skipped_existing: bool = False
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_elements(self) -> int:
        return self.text_chunks + self.tables + self.figures

    @classmethod
    def from_elements(
        cls,
        document_id: str,
        filename: str,
        pages: int,
        elements: list[Element],
        **kwargs: Any,
    ) -> Self:
        counts = dict.fromkeys(ElementKind, 0)
        for element in elements:
            counts[element.kind] += 1
        return cls(
            document_id=document_id,
            filename=filename,
            pages=pages,
            text_chunks=counts[ElementKind.TEXT],
            tables=counts[ElementKind.TABLE],
            figures=counts[ElementKind.FIGURE],
            **kwargs,
        )


def compute_document_id(pdf_path: Path | str) -> str:
    """Content-addressed document id.

    Hashing the bytes (not the filename) makes re-ingestion idempotent: the
    same PDF uploaded twice under different names maps to the same id, so we
    can skip the expensive summarisation pass.
    """
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]
