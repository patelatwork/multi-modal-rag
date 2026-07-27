"""Persistence: vector index and element store."""

from __future__ import annotations

from multimodal_rag.storage.docstore import DocumentRecord, ElementStore
from multimodal_rag.storage.metadata import sanitize_metadata
from multimodal_rag.storage.vector_store import MultiVectorIndex, get_index

__all__ = [
    "DocumentRecord",
    "ElementStore",
    "MultiVectorIndex",
    "get_index",
    "sanitize_metadata",
]
