"""PDF extraction backends."""

from __future__ import annotations

from multimodal_rag.config import ExtractorBackend, Settings, get_settings
from multimodal_rag.exceptions import ConfigurationError
from multimodal_rag.extractors.base import ExtractionResult, PDFExtractor
from multimodal_rag.extractors.pymupdf_backend import PyMuPDFExtractor
from multimodal_rag.extractors.unstructured_backend import (
    UnstructuredExtractor,
    missing_system_dependencies,
)

__all__ = [
    "ExtractionResult",
    "PDFExtractor",
    "PyMuPDFExtractor",
    "UnstructuredExtractor",
    "get_extractor",
    "missing_system_dependencies",
]


def get_extractor(
    settings: Settings | None = None,
    backend: ExtractorBackend | str | None = None,
) -> PDFExtractor:
    """Build the configured extraction backend.

    ``backend`` overrides the configured default, which lets a single request
    opt into OCR without changing process-wide settings.
    """
    settings = settings or get_settings()
    choice = ExtractorBackend(backend) if backend else settings.extractor_backend

    match choice:
        case ExtractorBackend.PYMUPDF:
            return PyMuPDFExtractor(settings)
        case ExtractorBackend.UNSTRUCTURED:
            return UnstructuredExtractor(settings)
        case _:  # pragma: no cover - guarded by the enum
            msg = f"Unknown extractor backend: {choice}"
            raise ConfigurationError(msg)
