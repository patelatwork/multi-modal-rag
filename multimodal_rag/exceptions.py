"""Exception hierarchy.

Every failure the application raises deliberately derives from
:class:`MultimodalRagError`, so the API layer can map domain failures to HTTP
status codes without catching bare ``Exception``.
"""

from __future__ import annotations


class MultimodalRagError(Exception):
    """Base class for all application errors."""


class ConfigurationError(MultimodalRagError):
    """A required setting is missing or invalid."""


class ExtractionError(MultimodalRagError):
    """A PDF could not be parsed."""


class UnsupportedFileError(ExtractionError):
    """The uploaded file is not a PDF, or is corrupt."""


class LLMError(MultimodalRagError):
    """The chat model failed after exhausting retries."""


class EmbeddingError(MultimodalRagError):
    """Embeddings could not be computed."""


class StorageError(MultimodalRagError):
    """The vector store or docstore rejected an operation."""


class DocumentNotFoundError(MultimodalRagError):
    """A requested document or asset is not present in the index."""
