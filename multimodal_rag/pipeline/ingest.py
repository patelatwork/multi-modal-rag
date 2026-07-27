"""Ingestion: PDF in, searchable multimodal index out."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from multimodal_rag.config import ExtractorBackend, Settings, get_settings
from multimodal_rag.domain import IngestionReport, compute_document_id
from multimodal_rag.exceptions import UnsupportedFileError
from multimodal_rag.extractors import get_extractor
from multimodal_rag.logging_config import get_logger
from multimodal_rag.pipeline.summarize import ProgressCallback, summarize_elements
from multimodal_rag.storage.vector_store import MultiVectorIndex, get_index

logger = get_logger(__name__)

_PDF_MAGIC = b"%PDF-"


def validate_pdf(path: Path) -> None:
    """Reject non-PDFs before spending time (and tokens) on them."""
    if not path.is_file():
        msg = f"File not found: {path}"
        raise UnsupportedFileError(msg)
    if path.suffix.lower() != ".pdf":
        suffix = path.suffix or "no extension"
        msg = f"Only PDF files are supported, got '{suffix}'"
        raise UnsupportedFileError(msg)
    # Trusting the extension is how a renamed .zip ends up inside the parser.
    with path.open("rb") as handle:
        if handle.read(len(_PDF_MAGIC)) != _PDF_MAGIC:
            msg = f"{path.name} is not a valid PDF (bad file header)"
            raise UnsupportedFileError(msg)


def ingest_pdf(
    pdf_path: Path | str,
    *,
    settings: Settings | None = None,
    index: MultiVectorIndex | None = None,
    backend: ExtractorBackend | str | None = None,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> IngestionReport:
    """Extract, summarise and index one PDF.

    Ingestion is idempotent: the document id is a hash of the file's bytes, so
    re-uploading the same PDF short-circuits instead of paying for every vision
    call again. Pass ``force=True`` to rebuild it anyway.
    """
    settings = settings or get_settings()
    index = index or get_index(settings)
    pdf_path = Path(pdf_path)
    started = time.perf_counter()

    validate_pdf(pdf_path)
    document_id = compute_document_id(pdf_path)

    if index.has_document(document_id) and not force:
        logger.info("%s is already indexed (%s); skipping", pdf_path.name, document_id)
        return IngestionReport(
            document_id=document_id,
            filename=pdf_path.name,
            pages=0,
            skipped_existing=True,
            elapsed_seconds=time.perf_counter() - started,
        )

    if force and index.has_document(document_id):
        index.delete_document(document_id)

    extractor = get_extractor(settings, backend)
    image_dir = settings.images_dir / document_id
    result = extractor.extract(pdf_path, document_id, image_dir)

    logger.info(
        "Summarising %s elements from %s (%s figures/tables need vision calls)",
        len(result.elements),
        pdf_path.name,
        sum(1 for element in result.elements if element.kind.is_visual),
    )
    indexed = summarize_elements(result.elements, settings=settings, on_progress=on_progress)

    stored_pdf = _archive_pdf(pdf_path, document_id, settings)
    index.register_document(
        document_id=document_id,
        filename=pdf_path.name,
        pages=result.pages,
        backend=result.backend,
        stored_pdf_path=str(stored_pdf) if stored_pdf else None,
    )
    index.add_elements(indexed)

    report = IngestionReport.from_elements(
        document_id=document_id,
        filename=pdf_path.name,
        pages=result.pages,
        elements=[item.element for item in indexed],
        elapsed_seconds=time.perf_counter() - started,
        warnings=result.warnings,
    )
    logger.info(
        "Ingested %s in %.1fs: %s text, %s tables, %s figures",
        pdf_path.name,
        report.elapsed_seconds,
        report.text_chunks,
        report.tables,
        report.figures,
    )
    return report


def _archive_pdf(pdf_path: Path, document_id: str, settings: Settings) -> Path | None:
    """Keep a copy of the source PDF so the UI can link back to the original."""
    target_dir = settings.extraction_dir / "sources"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{document_id}.pdf"
    if target.exists():
        return target
    try:
        shutil.copy2(pdf_path, target)
    except OSError as error:  # archiving is a convenience, not a requirement
        logger.warning("Could not archive %s: %s", pdf_path.name, error)
        return None
    return target


def ingest_directory(
    directory: Path | str,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> list[IngestionReport]:
    """Ingest every PDF in a directory, continuing past individual failures."""
    settings = settings or get_settings()
    index = get_index(settings)
    directory = Path(directory)

    reports: list[IngestionReport] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        try:
            reports.append(ingest_pdf(pdf_path, settings=settings, index=index, force=force))
        except Exception as error:
            logger.error("Failed to ingest %s: %s", pdf_path.name, error)
            reports.append(
                IngestionReport(
                    document_id="",
                    filename=pdf_path.name,
                    pages=0,
                    warnings=[str(error)],
                )
            )
    return reports
