"""Application configuration.

All settings are loaded from environment variables (or a local ``.env`` file)
and validated once at import time via :func:`get_settings`.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExtractorBackend(StrEnum):
    """Which PDF parsing engine to use."""

    PYMUPDF = "pymupdf"
    UNSTRUCTURED = "unstructured"


class EmbeddingProvider(StrEnum):
    """Where sentence embeddings are computed."""

    LOCAL = "local"
    HUGGINGFACE_API = "hf_api"


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ paths
    project_root: Path = PROJECT_ROOT
    input_pdf_dir: Path = Field(default=PROJECT_ROOT / "data" / "pdfs")
    extraction_dir: Path = Field(default=PROJECT_ROOT / "data" / "extracted")
    vector_store_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    docstore_dir: Path = Field(default=PROJECT_ROOT / "data" / "docstore")

    # -------------------------------------------------------------- chat model
    groq_api_key: str | None = None
    groq_model: str = "qwen/qwen3.6-27b"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1536, gt=0)
    """Counts toward the provider's per-request token budget, not just the reply."""
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=4, ge=0)
    llm_max_backoff_seconds: float = Field(default=75.0, gt=0)
    """Upper bound on a single backoff sleep. Must exceed a provider rate-limit
    window (Groq's free tier resets per minute) or retries expire too early."""

    llm_min_request_interval: float = Field(default=0.0, ge=0.0)
    """Seconds to space consecutive chat calls. Raise on rate-limited tiers to
    trade throughput for far fewer 429s during ingestion."""

    # Groq reasoning models emit <think> blocks unless this is "none".
    llm_reasoning_effort: str | None = "none"

    # -------------------------------------------------------------- embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_api_token: str | None = None
    embedding_batch_size: int = Field(default=32, gt=0)

    # -------------------------------------------------------------- extraction
    extractor_backend: ExtractorBackend = ExtractorBackend.PYMUPDF
    render_dpi: int = Field(default=200, ge=72, le=600)
    """DPI used when rasterising figure and table regions for the vision model."""

    min_region_area_ratio: float = Field(default=0.012, ge=0.0, le=1.0)
    """Regions smaller than this fraction of the page are discarded (logos, rules)."""

    max_region_area_ratio: float = Field(default=0.95, ge=0.0, le=1.0)
    """Regions larger than this are treated as page backgrounds, not figures."""

    region_merge_gap: float = Field(default=14.0, ge=0.0)
    """PDF points; nearby image/vector boxes closer than this are merged into one figure."""

    ocr_enabled: bool = False
    """Only meaningful for the ``unstructured`` backend (requires tesseract)."""

    # ---------------------------------------------------------------- chunking
    chunk_size: int = Field(default=1400, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # --------------------------------------------------------------- retrieval
    collection_name: str = "multimodal_rag"
    retrieval_k: int = Field(default=6, gt=0)

    lexical_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    """Share of the hybrid ranking taken by verbatim overlap rather than
    embedding similarity. Set to 0 for pure dense retrieval; raise it if queries
    routinely name specific figures, tables or identifiers."""

    rerank_fetch_multiplier: int = Field(default=4, ge=1)
    """How many extra candidates to pull from the vector store before
    re-ranking. Re-ranking can only reorder what dense search returned."""
    max_images_per_answer: int = Field(default=2, ge=0)
    """Cap on images sent to the LLM per answer.

    Sized for a rate-limited free tier: images dominate token cost, and Groq's
    free tier allows only 8000 tokens per request. On a paid tier raise this
    (and ``max_context_chars``) for richer answers. Requests that still come
    back too large are shrunk and retried automatically."""

    max_context_chars: int = Field(default=6_000, gt=0)

    # --------------------------------------------------------------- summaries
    summarize_concurrency: int = Field(default=4, gt=0)
    summarize_text_chunks: bool = False
    """Off by default: prose already embeds well verbatim, and summarising every
    chunk doubles ingestion cost for a marginal retrieval gain. Figures and
    tables are always summarised -- for them the summary *is* the only text."""

    image_max_dimension: int = Field(default=900, ge=256)
    """Rendered crops are downscaled to this before being sent to the model.
    Image tokens scale with area, so this is the main lever on cost and on how
    often a rate limit is hit; 900px keeps chart labels and table cells legible."""

    image_jpeg_quality: int = Field(default=85, ge=1, le=95)

    # --------------------------------------------------------------------- api
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # NoDecode: pydantic-settings would otherwise JSON-parse the raw env value
    # before our validator sees it, so a plain comma-separated list would fail.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    max_upload_mb: int = Field(default=50, gt=0)

    # ------------------------------------------------------------------ system
    log_level: str = "INFO"

    # ---------------------------------------------------------------- plumbing
    @field_validator(
        "input_pdf_dir",
        "extraction_dir",
        "vector_store_dir",
        "docstore_dir",
        mode="after",
    )
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Make relative paths (e.g. ``./data/chroma``) stable regardless of cwd."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            msg = "chunk_overlap must be smaller than chunk_size"
            raise ValueError(msg)
        if self.min_region_area_ratio >= self.max_region_area_ratio:
            msg = "min_region_area_ratio must be smaller than max_region_area_ratio"
            raise ValueError(msg)
        return self

    @property
    def images_dir(self) -> Path:
        """Where rasterised figure and table crops are written."""
        return self.extraction_dir / "images"

    def ensure_directories(self) -> None:
        for directory in (
            self.input_pdf_dir,
            self.extraction_dir,
            self.images_dir,
            self.vector_store_dir,
            self.docstore_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton, creating data dirs on first use."""
    settings = Settings()
    settings.ensure_directories()
    return settings
