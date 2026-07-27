"""FastAPI application.

Exposes the pipelines over HTTP: upload a PDF, poll the ingestion job, ask
questions, and fetch the rendered figure or table behind any citation.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from multimodal_rag import __version__
from multimodal_rag.api.jobs import Job, JobRegistry
from multimodal_rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    IngestionResponse,
    JobResponse,
)
from multimodal_rag.config import Settings, get_settings
from multimodal_rag.exceptions import (
    ConfigurationError,
    DocumentNotFoundError,
    ExtractionError,
    LLMError,
    MultimodalRagError,
    UnsupportedFileError,
)
from multimodal_rag.logging_config import configure_logging, get_logger
from multimodal_rag.pipeline.ingest import ingest_pdf
from multimodal_rag.pipeline.query import answer_question
from multimodal_rag.storage.vector_store import MultiVectorIndex, get_index

logger = get_logger(__name__)

# Domain failures map onto status codes here rather than at each call site.
_STATUS_BY_ERROR: list[tuple[type[Exception], int]] = [
    (UnsupportedFileError, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
    (DocumentNotFoundError, status.HTTP_404_NOT_FOUND),
    (ConfigurationError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    # Literal, because Starlette renamed this constant and both spellings warn
    # on one version or the other.
    (ExtractionError, 422),
    (LLMError, status.HTTP_502_BAD_GATEWAY),
    (MultimodalRagError, status.HTTP_500_INTERNAL_SERVER_ERROR),
]


def _status_for(error: Exception) -> int:
    for error_type, code in _STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting Multimodal RAG API v%s", __version__)

    app.state.settings = settings
    app.state.index = get_index(settings)
    app.state.jobs = JobRegistry()
    try:
        yield
    finally:
        app.state.jobs.shutdown()
        logger.info("Shutting down")


app = FastAPI(
    title="Multimodal RAG API",
    version=__version__,
    summary="Question answering over PDFs, including their figures and tables.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(MultimodalRagError)
async def handle_domain_error(request: Request, error: MultimodalRagError) -> JSONResponse:
    code = _status_for(error)
    if code >= 500:
        logger.error("%s on %s: %s", type(error).__name__, request.url.path, error)
    return JSONResponse(
        status_code=code,
        content=ErrorResponse(detail=str(error), kind=type(error).__name__).model_dump(),
    )


# ---------------------------------------------------------------- dependencies
def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_app_index(request: Request) -> MultiVectorIndex:
    return request.app.state.index


def get_jobs(request: Request) -> JobRegistry:
    return request.app.state.jobs


# --------------------------------------------------------------------- routes
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health(
    settings: Settings = Depends(get_app_settings),
    index: MultiVectorIndex = Depends(get_app_index),
) -> HealthResponse:
    """Liveness plus a summary of what is configured and indexed."""
    detail: str | None = None
    state: str = "ok"
    documents = 0
    elements = 0

    try:
        documents = len(index.list_documents())
        elements = index.count_elements()
    except Exception as error:
        state = "degraded"
        detail = f"storage unavailable: {error}"

    if not settings.groq_api_key:
        state = "degraded"
        detail = "GROQ_API_KEY is not configured"

    return HealthResponse(
        status=state,  # type: ignore[arg-type]
        version=__version__,
        chat_model=settings.groq_model,
        embedding_provider=settings.embedding_provider.value,
        extractor_backend=settings.extractor_backend.value,
        documents=documents,
        elements=elements,
        detail=detail,
    )


@app.post(
    "/api/documents",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(...),
    force: bool = False,
    settings: Settings = Depends(get_app_settings),
    index: MultiVectorIndex = Depends(get_app_index),
    jobs: JobRegistry = Depends(get_jobs),
) -> JobResponse:
    """Accept a PDF and start ingesting it in the background."""
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf files are accepted",
        )

    upload_dir = Path(tempfile.mkdtemp(prefix="mmrag-upload-"))
    target = upload_dir / filename
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0

    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,  # literal: Starlette renamed this constant
                        detail=f"File exceeds the {settings.max_upload_mb} MB limit",
                    )
                handle.write(chunk)
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    def run(job: Job) -> None:
        try:
            report = ingest_pdf(
                target,
                settings=settings,
                index=index,
                force=force,
                on_progress=job.set_progress,
            )
            job.mark("succeeded", result=IngestionResponse.from_report(report))
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    return jobs.submit(filename, target, run).to_response()


@app.get("/api/jobs/{job_id}", response_model=JobResponse, tags=["documents"])
def get_job(job_id: str, jobs: JobRegistry = Depends(get_jobs)) -> JobResponse:
    """Poll an ingestion job."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job")
    return job.to_response()


@app.get("/api/documents", response_model=list[DocumentResponse], tags=["documents"])
def list_documents(index: MultiVectorIndex = Depends(get_app_index)) -> list[DocumentResponse]:
    """Every indexed document, newest first."""
    return [DocumentResponse.from_record(record) for record in index.list_documents()]


@app.delete(
    "/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"]
)
def delete_document(document_id: str, index: MultiVectorIndex = Depends(get_app_index)) -> None:
    """Remove a document from the index."""
    if not index.has_document(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown document")
    index.delete_document(document_id)


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(
    payload: ChatRequest,
    settings: Settings = Depends(get_app_settings),
    index: MultiVectorIndex = Depends(get_app_index),
) -> ChatResponse:
    """Answer a question, returning the figures and tables it relied on."""
    answer = answer_question(
        payload.question,
        settings=settings,
        index=index,
        k=payload.k,
        document_ids=payload.document_ids,
    )
    return ChatResponse.from_answer(answer)


@app.get("/api/elements/{element_id}/image", tags=["chat"])
def get_element_image(
    element_id: str,
    index: MultiVectorIndex = Depends(get_app_index),
) -> FileResponse:
    """Serve the rendered crop for a figure or table."""
    image_path = index.resolve_image(element_id)
    if image_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No image for this element"
        )

    # The path came from our own store, but resolve it against the configured
    # image root anyway so a poisoned row cannot read arbitrary files.
    root = index.settings.images_dir.resolve()
    resolved = Path(image_path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image unavailable")

    return FileResponse(
        resolved,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# --------------------------------------------------------------- static files
# Mounted last so it never shadows an /api route. Present only when a built
# frontend has been copied in (the Docker image does this), which lets a single
# container serve both the API and the UI. In development the Vite dev server
# handles the UI instead and this is skipped.
_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    logger.info("Serving the built frontend from %s", _STATIC_DIR)
