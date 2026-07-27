"""Embedding providers.

Two interchangeable backends, both returning a LangChain ``Embeddings``:

``local``
    ``sentence-transformers`` running in-process. The default, because it needs
    no API key, has no rate limit, works offline, and gives byte-identical
    vectors across runs -- which matters when a vector store is persisted and
    queried weeks later.

``hf_api``
    Hugging Face Inference API. Useful when the deployment target cannot afford
    the model download or the CPU time.

Switching providers changes the embedding space, so an index built with one
cannot be queried with the other. :func:`embedding_fingerprint` records which
was used and the ingest path refuses to mix them.
"""

from __future__ import annotations

import functools

from langchain_core.embeddings import Embeddings

from multimodal_rag.config import EmbeddingProvider, Settings, get_settings
from multimodal_rag.exceptions import ConfigurationError, EmbeddingError
from multimodal_rag.logging_config import get_logger

logger = get_logger(__name__)


def embedding_fingerprint(settings: Settings | None = None) -> str:
    """Stable identifier for the active embedding space."""
    settings = settings or get_settings()
    return f"{settings.embedding_provider.value}:{settings.embedding_model}"


def _build_local(settings: Settings) -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as error:  # pragma: no cover - optional dependency
        msg = (
            "Local embeddings need sentence-transformers. Run: "
            "pip install langchain-huggingface sentence-transformers"
        )
        raise ConfigurationError(msg) from error

    logger.info("Loading local embedding model %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={
            "batch_size": settings.embedding_batch_size,
            # Chroma ranks with cosine distance; unit vectors make those scores
            # comparable across documents.
            "normalize_embeddings": True,
        },
    )


def _build_hf_api(settings: Settings) -> Embeddings:
    if not settings.hf_api_token:
        msg = "HF_API_TOKEN is required when EMBEDDING_PROVIDER=hf_api"
        raise ConfigurationError(msg)

    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
    except ImportError as error:  # pragma: no cover - optional dependency
        msg = "Run: pip install langchain-huggingface"
        raise ConfigurationError(msg) from error

    logger.info("Using Hugging Face Inference API for embeddings (%s)", settings.embedding_model)
    return HuggingFaceEndpointEmbeddings(
        model=settings.embedding_model,
        huggingfacehub_api_token=settings.hf_api_token,
    )


@functools.lru_cache(maxsize=1)
def _default_embeddings() -> Embeddings:
    return _build(get_settings())


def _build(settings: Settings) -> Embeddings:
    match settings.embedding_provider:
        case EmbeddingProvider.LOCAL:
            return _build_local(settings)
        case EmbeddingProvider.HUGGINGFACE_API:
            return _build_hf_api(settings)
        case _:  # pragma: no cover - guarded by the enum
            msg = f"Unknown embedding provider: {settings.embedding_provider}"
            raise ConfigurationError(msg)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the configured embeddings, caching the default instance.

    The local backend holds a loaded transformer, so reusing one instance
    avoids re-reading the weights on every request.
    """
    if settings is None:
        return _default_embeddings()
    return _build(settings)


def embedding_dimension(settings: Settings | None = None) -> int:
    """Probe the embedding size. Also doubles as a connectivity check."""
    try:
        return len(get_embeddings(settings).embed_query("dimension probe"))
    except Exception as error:
        msg = f"Embedding provider is not usable: {error}"
        raise EmbeddingError(msg) from error
