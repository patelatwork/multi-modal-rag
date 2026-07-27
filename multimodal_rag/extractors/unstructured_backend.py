"""Unstructured extraction backend (opt-in).

Worth choosing over the PyMuPDF default when the PDFs are **scanned**: the
``hi_res`` strategy runs a layout-detection model and OCR, which recovers text
PyMuPDF cannot see because there is none in the file.

The cost is system dependencies. ``poppler`` and ``tesseract`` must be
installed and on ``PATH``; this module checks for them up front and raises a
clear error rather than failing deep inside a worker.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from multimodal_rag.config import Settings, get_settings
from multimodal_rag.domain import Element, ElementKind
from multimodal_rag.exceptions import ExtractionError, UnsupportedFileError
from multimodal_rag.extractors.base import ExtractionResult
from multimodal_rag.logging_config import get_logger

logger = get_logger(__name__)

_REQUIRED_BINARIES = {
    "pdfinfo": "poppler (pdfinfo/pdftoppm)",
    "tesseract": "tesseract-ocr",
}


def missing_system_dependencies(*, require_ocr: bool = True) -> list[str]:
    """Return human-readable names of the binaries unstructured needs but lacks."""
    missing: list[str] = []
    for binary, label in _REQUIRED_BINARIES.items():
        if binary == "tesseract" and not require_ocr:
            continue
        if shutil.which(binary) is None:
            missing.append(label)
    return missing


class UnstructuredExtractor:
    """Wraps ``unstructured.partition.pdf.partition_pdf``."""

    name = "unstructured"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def extract(self, pdf_path: Path, document_id: str, image_dir: Path) -> ExtractionResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            msg = f"PDF not found: {pdf_path}"
            raise UnsupportedFileError(msg)

        missing = missing_system_dependencies(require_ocr=self.settings.ocr_enabled)
        if missing:
            msg = (
                "The 'unstructured' backend needs these system packages on PATH: "
                f"{', '.join(missing)}. Install them, or set EXTRACTOR_BACKEND=pymupdf "
                "to use the dependency-free default backend."
            )
            raise ExtractionError(msg)

        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as error:  # pragma: no cover - optional dependency
            msg = "unstructured is not installed. Run: pip install 'unstructured[all-docs]'"
            raise ExtractionError(msg) from error

        image_dir.mkdir(parents=True, exist_ok=True)

        try:
            raw_elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res" if self.settings.ocr_enabled else "fast",
                infer_table_structure=True,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_output_dir=str(image_dir),
                # Chunking here is what keeps summarisation affordable: without
                # it every heading and list item becomes its own LLM call.
                chunking_strategy="by_title",
                max_characters=self.settings.chunk_size,
                combine_text_under_n_chars=self.settings.chunk_size // 4,
                new_after_n_chars=int(self.settings.chunk_size * 0.9),
            )
        except Exception as error:
            msg = f"unstructured failed to parse {pdf_path.name}: {error}"
            raise ExtractionError(msg) from error

        elements: list[Element] = []
        pages = 0

        for index, raw in enumerate(raw_elements):
            metadata = _metadata_to_dict(getattr(raw, "metadata", None))
            page_number = metadata.get("page_number")
            if isinstance(page_number, int):
                pages = max(pages, page_number)

            category = type(raw).__name__
            if category == "Table":
                kind = ElementKind.TABLE
                text = metadata.get("text_as_html") or getattr(raw, "text", "") or ""
            elif category == "Image":
                kind = ElementKind.FIGURE
                text = getattr(raw, "text", "") or ""
            else:
                kind = ElementKind.TEXT
                text = getattr(raw, "text", "") or ""

            image_path = metadata.get("image_path")
            text = str(text).strip()
            if not text and not image_path:
                continue

            elements.append(
                Element(
                    element_id=f"{document_id}:{kind.value}:{index}",
                    document_id=document_id,
                    kind=kind,
                    text=text,
                    page_number=page_number if isinstance(page_number, int) else None,
                    image_path=Path(image_path) if image_path else None,
                    source_pdf=str(pdf_path),
                    extra={"category": category},
                )
            )

        if not elements:
            msg = f"No content could be extracted from {pdf_path.name}"
            raise ExtractionError(msg)

        logger.info("Extracted %s elements from %s via %s", len(elements), pdf_path.name, self.name)
        return ExtractionResult(
            document_id=document_id,
            source_pdf=pdf_path,
            pages=pages,
            elements=elements,
            backend=self.name,
        )


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if hasattr(metadata, "to_dict"):
        return dict(metadata.to_dict())
    if isinstance(metadata, dict):
        return dict(metadata)
    if hasattr(metadata, "__dict__"):
        return {key: value for key, value in vars(metadata).items() if value is not None}
    return {}
