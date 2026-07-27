"""Request and response models for the HTTP API.

Kept separate from :mod:`multimodal_rag.domain` so the wire format can evolve
without dragging the pipelines along, and so OpenAPI documents exactly what a
client sees.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from multimodal_rag.domain import Answer, ElementKind, IngestionReport, RetrievedSource
from multimodal_rag.storage.docstore import DocumentRecord


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    chat_model: str
    embedding_provider: str
    extractor_backend: str
    documents: int
    elements: int
    detail: str | None = None


class SourceResponse(BaseModel):
    """One piece of evidence behind an answer."""

    id: str = Field(description="Stable element id, also used to fetch its image")
    label: str = Field(description="Citation label used in the answer text, e.g. 'S1'")
    document_id: str
    kind: ElementKind
    page_number: int | None = None
    caption: str | None = None
    text: str
    score: float | None = None
    image_url: str | None = Field(
        default=None, description="Set for figures and tables that have a rendered crop"
    )
    table_markdown: str | None = Field(
        default=None, description="Markdown transcription, when the source is a table"
    )

    @classmethod
    def from_source(cls, source: RetrievedSource, position: int) -> SourceResponse:
        return cls(
            id=source.element_id,
            label=f"S{position}",
            document_id=source.document_id,
            kind=source.kind,
            page_number=source.page_number,
            caption=source.caption,
            text=source.text,
            score=source.score,
            image_url=(f"/api/elements/{source.element_id}/image" if source.has_image else None),
            table_markdown=_extract_table(source),
        )


def _extract_table(source: RetrievedSource) -> str | None:
    """Pull the Markdown table out of a table source's indexed text."""
    if source.kind is not ElementKind.TABLE:
        return None
    marker = source.text.find("|")
    if marker == -1:
        return None
    candidate = source.text[marker:].strip()
    # A real Markdown table needs a header row and a separator row.
    lines = [line for line in candidate.splitlines() if line.strip().startswith("|")]
    return "\n".join(lines) if len(lines) >= 2 else None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these documents"
    )
    k: int | None = Field(default=None, ge=1, le=25, description="Number of sources to retrieve")


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    images_sent_to_model: int
    elapsed_seconds: float

    @classmethod
    def from_answer(cls, answer: Answer) -> ChatResponse:
        return cls(
            question=answer.question,
            answer=answer.text,
            sources=[
                SourceResponse.from_source(source, position)
                for position, source in enumerate(answer.sources, start=1)
            ],
            images_sent_to_model=answer.images_sent_to_model,
            elapsed_seconds=round(answer.elapsed_seconds, 2),
        )


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    element_count: int
    backend: str | None = None
    created_at: str

    @classmethod
    def from_record(cls, record: DocumentRecord) -> DocumentResponse:
        return cls(
            document_id=record.document_id,
            filename=record.filename,
            pages=record.pages,
            element_count=record.element_count,
            backend=record.backend,
            created_at=record.created_at,
        )


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    text_chunks: int
    tables: int
    figures: int
    skipped_existing: bool
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_report(cls, report: IngestionReport) -> IngestionResponse:
        return cls(
            document_id=report.document_id,
            filename=report.filename,
            pages=report.pages,
            text_chunks=report.text_chunks,
            tables=report.tables,
            figures=report.figures,
            skipped_existing=report.skipped_existing,
            elapsed_seconds=round(report.elapsed_seconds, 2),
            warnings=report.warnings,
        )


JobStatus = Literal["pending", "running", "succeeded", "failed"]


class JobResponse(BaseModel):
    """Progress of a background ingestion.

    Parsing a long PDF costs one vision call per figure, so ingestion runs
    detached and the client polls this instead of holding a request open.
    """

    job_id: str
    status: JobStatus
    filename: str
    processed: int = 0
    total: int = 0
    result: IngestionResponse | None = None
    error: str | None = None

    @property
    def percent(self) -> int:
        return int(100 * self.processed / self.total) if self.total else 0


class ErrorResponse(BaseModel):
    detail: str
    kind: str
