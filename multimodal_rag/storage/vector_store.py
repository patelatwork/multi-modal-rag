from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from multimodal_rag.config import AppConfig, get_settings
from multimodal_rag.extractors.unstructured_loader import ExtractedElement
from multimodal_rag.models.llm import get_embeddings


@dataclass(slots=True)
class IndexedItem:
    doc_id: str
    kind: str
    summary: str
    raw_content: str
    metadata: dict[str, object]


def get_vector_store(settings: AppConfig | None = None) -> Chroma:
    settings = settings or get_settings()
    embeddings = get_embeddings(settings)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.vector_store_dir),
    )


def get_retriever(settings: AppConfig | None = None) -> MultiVectorRetriever:
    settings = settings or get_settings()
    vector_store = get_vector_store(settings)
    docstore = create_kv_docstore(LocalFileStore(settings.docstore_dir))
    return MultiVectorRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={"k": settings.retrieval_k},
    )


def build_indexed_items(extracted_elements: list[ExtractedElement], summaries: dict[str, str]) -> list[IndexedItem]:
    indexed_items: list[IndexedItem] = []
    for element in extracted_elements:
        indexed_items.append(
            IndexedItem(
                doc_id=element.doc_id,
                kind=element.kind,
                summary=summaries.get(element.doc_id, element.content),
                raw_content=element.content,
                metadata=element.metadata,
            )
        )
    return indexed_items


def ingest_items(items: list[IndexedItem], settings: AppConfig | None = None) -> MultiVectorRetriever:
    settings = settings or get_settings()
    vector_store = get_vector_store(settings)
    docstore = create_kv_docstore(LocalFileStore(settings.docstore_dir))

    summary_documents = [
        Document(page_content=item.summary, metadata={**item.metadata, "doc_id": item.doc_id, "kind": item.kind})
        for item in items
    ]
    raw_documents = [
        Document(page_content=item.raw_content, metadata={**item.metadata, "doc_id": item.doc_id, "kind": item.kind})
        for item in items
    ]
    doc_ids = [item.doc_id for item in items]

    if summary_documents:
        vector_store.add_documents(summary_documents, ids=doc_ids)
        if hasattr(vector_store, "persist"):
            vector_store.persist()

    if raw_documents:
        docstore.mset(list(zip(doc_ids, raw_documents)))

    return MultiVectorRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={"k": settings.retrieval_k},
    )