"""Multi-vector index over Chroma plus the SQLite element store.

Implements the multi-vector retrieval pattern explicitly: what gets **embedded**
is a compact summary of each element, while what gets **returned** is the full
element with its rendered image. That indirection is the reason a chart can be
found at all -- an image has no text to embed, so its model-written description
stands in for it at search time, and the image itself is handed to the answering
model afterwards.
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from multimodal_rag.config import Settings, get_settings
from multimodal_rag.domain import ElementKind, IndexedElement, RetrievedSource
from multimodal_rag.exceptions import StorageError
from multimodal_rag.logging_config import get_logger
from multimodal_rag.models.embeddings import embedding_fingerprint, get_embeddings
from multimodal_rag.storage.docstore import DocumentRecord, ElementStore
from multimodal_rag.storage.metadata import sanitize_metadata
from multimodal_rag.storage.ranking import Candidate, extract_references, hybrid_rank

logger = get_logger(__name__)

# Chroma rejects oversized batches; well under the limit keeps memory flat too.
_ADD_BATCH_SIZE = 128


class MultiVectorIndex:
    """Write and query side of the index."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.store = ElementStore(self.settings.docstore_dir)
        self._vectors: Chroma | None = None

    @property
    def vectors(self) -> Chroma:
        """Lazily built so importing this module never loads an embedding model."""
        if self._vectors is None:
            self._vectors = Chroma(
                collection_name=self.settings.collection_name,
                embedding_function=get_embeddings(self.settings),
                persist_directory=str(self.settings.vector_store_dir),
                # Cosine matches the normalised sentence-transformer vectors;
                # Chroma's default L2 would rank them less meaningfully.
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._vectors

    # ----------------------------------------------------------------- writing
    def add_elements(self, items: list[IndexedElement]) -> int:
        """Embed each summary and persist the full elements. Returns count added."""
        if not items:
            return 0

        documents: list[Document] = []
        ids: list[str] = []

        for item in items:
            element = item.element
            content = element.retrieval_text(item.summary)
            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata=sanitize_metadata(
                        {
                            "element_id": element.element_id,
                            "document_id": element.document_id,
                            "kind": element.kind.value,
                            "page_number": element.page_number,
                            "has_image": element.has_image,
                            "caption": element.caption,
                            "source_pdf": element.source_pdf,
                        }
                    ),
                )
            )
            ids.append(element.element_id)

        if not documents:
            return 0

        try:
            for start in range(0, len(documents), _ADD_BATCH_SIZE):
                chunk = documents[start : start + _ADD_BATCH_SIZE]
                chunk_ids = ids[start : start + _ADD_BATCH_SIZE]
                # Explicit ids make re-ingesting a document an upsert instead of
                # silently duplicating every chunk.
                self.vectors.add_documents(chunk, ids=chunk_ids)
        except Exception as error:
            msg = f"Could not write to the vector store: {error}"
            raise StorageError(msg) from error

        self.store.put_elements(items)
        logger.info("Indexed %s elements", len(documents))
        return len(documents)

    def register_document(
        self,
        *,
        document_id: str,
        filename: str,
        pages: int,
        backend: str,
        stored_pdf_path: str | None = None,
    ) -> None:
        self.store.upsert_document(
            document_id=document_id,
            filename=filename,
            pages=pages,
            backend=backend,
            embedding_fingerprint=embedding_fingerprint(self.settings),
            stored_pdf_path=stored_pdf_path,
        )

    def delete_document(self, document_id: str) -> int:
        """Remove a document from both halves of the index."""
        try:
            self.vectors.delete(where={"document_id": document_id})
        except Exception as error:  # a missing collection is not fatal here
            logger.warning("Vector delete for %s failed: %s", document_id, error)
        removed = self.store.delete_document(document_id)
        logger.info("Deleted document %s (%s elements)", document_id, removed)
        return removed

    # ----------------------------------------------------------------- reading
    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        document_ids: list[str] | None = None,
        kinds: list[ElementKind] | None = None,
    ) -> list[RetrievedSource]:
        """Search the index, returning hydrated, display-ready sources.

        Runs a hybrid ranking: the vector store is over-fetched, then candidates
        are re-scored on dense similarity *and* verbatim overlap. Without the
        lexical half, "what does Figure 13 show" reliably returns Figures 5, 7
        and 10 -- their captions are near-identical in embedding space.
        """
        k = k or self.settings.retrieval_k

        conditions: list[dict] = []
        if document_ids:
            conditions.append({"document_id": {"$in": document_ids}})
        if kinds:
            conditions.append({"kind": {"$in": [kind.value for kind in kinds]}})

        where: dict | None = None
        if len(conditions) == 1:
            where = conditions[0]
        elif conditions:
            where = {"$and": conditions}

        fetch_k = max(k * self.settings.rerank_fetch_multiplier, k)
        try:
            hits = self.vectors.similarity_search_with_relevance_scores(
                query, k=fetch_k, filter=where
            )
        except Exception as error:
            msg = f"Vector search failed: {error}"
            raise StorageError(msg) from error

        if not hits:
            return []

        hits = self._rerank(query, hits, k)

        element_ids = [
            str(document.metadata.get("element_id"))
            for document, _ in hits
            if document.metadata.get("element_id")
        ]
        stored = self.store.get_elements(element_ids)

        sources: list[RetrievedSource] = []
        for document, score in hits:
            element_id = str(document.metadata.get("element_id", ""))
            record = stored.get(element_id)

            if record is None:
                # The vector row outlived its element (interrupted delete).
                # Fall back to the embedded summary so the answer still has text.
                logger.debug("No stored element for %s; using indexed summary", element_id)
                sources.append(
                    RetrievedSource(
                        element_id=element_id,
                        document_id=str(document.metadata.get("document_id", "")),
                        kind=ElementKind(str(document.metadata.get("kind", "text"))),
                        text=document.page_content,
                        page_number=_as_int(document.metadata.get("page_number")),
                        score=score,
                    )
                )
                continue

            element, summary = record
            sources.append(
                RetrievedSource(
                    element_id=element.element_id,
                    document_id=element.document_id,
                    kind=element.kind,
                    # For a figure the summary is the only text there is; for
                    # prose the original wording beats a lossy summary.
                    text=(summary if element.kind.is_visual else element.text) or summary,
                    page_number=element.page_number,
                    caption=element.caption,
                    image_path=element.image_path,
                    score=score,
                    source_pdf=element.source_pdf,
                )
            )

        return self._promote_referenced(query, sources, k, document_ids)

    def _promote_referenced(
        self,
        query: str,
        sources: list[RetrievedSource],
        k: int,
        document_ids: list[str] | None,
    ) -> list[RetrievedSource]:
        """Put explicitly named figures and tables at the front of the results.

        When a question says "Figure 13", that element is the answer whether or
        not embedding similarity agreed -- so it is looked up by caption and
        promoted, displacing the weakest dense hit.
        """
        references = extract_references(query)
        if not references:
            return sources

        matched = self.store.find_by_caption_reference(references, document_ids)
        already = {source.element_id for source in sources}
        missing = [element_id for element_id in matched if element_id not in already]
        if not missing:
            return sources

        stored = self.store.get_elements(missing)
        promoted: list[RetrievedSource] = []
        for element_id in missing:
            record = stored.get(element_id)
            if record is None:
                continue
            element, summary = record
            promoted.append(
                RetrievedSource(
                    element_id=element.element_id,
                    document_id=element.document_id,
                    kind=element.kind,
                    text=(summary if element.kind.is_visual else element.text) or summary,
                    page_number=element.page_number,
                    caption=element.caption,
                    image_path=element.image_path,
                    score=None,  # matched by reference, not by similarity
                    source_pdf=element.source_pdf,
                )
            )

        if promoted:
            logger.info("Promoted %s element(s) matching %s", len(promoted), sorted(references))
        return (promoted + sources)[:k]

    def _rerank(
        self, query: str, hits: list[tuple[Document, float]], k: int
    ) -> list[tuple[Document, float]]:
        """Blend dense scores with lexical overlap and keep the best ``k``."""
        if self.settings.lexical_weight <= 0:
            return hits[:k]

        candidates = [
            Candidate(key=str(index), text=document.page_content, dense_score=score)
            for index, (document, score) in enumerate(hits)
        ]
        ranked = hybrid_rank(
            query, candidates, top_k=k, lexical_weight=self.settings.lexical_weight
        )
        # Report the original relevance score, not the blended one: the blend is
        # only meaningful relative to this candidate set.
        return [
            (hits[int(candidate.key)][0], hits[int(candidate.key)][1]) for candidate, _ in ranked
        ]

    def has_document(self, document_id: str) -> bool:
        return self.store.has_document(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def count_elements(self, document_id: str | None = None) -> int:
        return self.store.count_elements(document_id)

    def resolve_image(self, element_id: str) -> Path | None:
        """Path to an element's rendered crop, if it has one."""
        record = self.store.get_elements([element_id]).get(element_id)
        if record is None:
            return None
        element, _ = record
        return element.image_path if element.has_image else None


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


_index: MultiVectorIndex | None = None


def get_index(settings: Settings | None = None) -> MultiVectorIndex:
    """Process-wide index singleton (holds an open SQLite connection)."""
    global _index
    if settings is not None:
        return MultiVectorIndex(settings)
    if _index is None:
        _index = MultiVectorIndex()
    return _index
