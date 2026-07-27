"""Chat and embedding model access."""

from __future__ import annotations

from multimodal_rag.models.embeddings import (
    embedding_dimension,
    embedding_fingerprint,
    get_embeddings,
)
from multimodal_rag.models.llm import VisionChatModel, get_chat_model

__all__ = [
    "VisionChatModel",
    "embedding_dimension",
    "embedding_fingerprint",
    "get_chat_model",
    "get_embeddings",
]
