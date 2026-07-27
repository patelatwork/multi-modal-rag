"""Extraction backend contract.

A backend turns a PDF into a flat list of :class:`~multimodal_rag.domain.Element`
objects. Everything downstream (summarisation, indexing, retrieval) is written
against this contract, so backends are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from multimodal_rag.domain import Element


@dataclass(slots=True)
class ExtractionResult:
    """Everything one backend learned about one PDF."""

    document_id: str
    source_pdf: Path
    pages: int
    elements: list[Element] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    backend: str = ""


@runtime_checkable
class PDFExtractor(Protocol):
    """Structural type implemented by every extraction backend."""

    name: str

    def extract(self, pdf_path: Path, document_id: str, image_dir: Path) -> ExtractionResult:
        """Parse ``pdf_path``, writing any rendered crops under ``image_dir``."""
        ...
