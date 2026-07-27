"""Storage tests: metadata coercion, the element store, and the index."""

from __future__ import annotations

from pathlib import Path

import pytest

from multimodal_rag.config import Settings
from multimodal_rag.domain import Element, ElementKind, IndexedElement
from multimodal_rag.storage.docstore import ElementStore
from multimodal_rag.storage.metadata import sanitize_metadata, sanitize_value
from multimodal_rag.storage.vector_store import MultiVectorIndex


class TestMetadataSanitization:
    """Chroma rejects non-scalar metadata; unsanitised PDF metadata always has some."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("text", "text"),
            (42, 42),
            (3.14, 3.14),
            (True, True),
            (None, None),
            ("   ", None),
            ("  padded  ", "padded"),
        ],
    )
    def test_scalars(self, value, expected) -> None:
        assert sanitize_value(value) == expected

    def test_lists_become_json(self) -> None:
        assert sanitize_value(["en", "fr"]) == '["en", "fr"]'

    def test_dicts_become_json(self) -> None:
        assert sanitize_value({"x": 1}) == '{"x": 1}'

    def test_long_values_are_truncated(self) -> None:
        assert len(sanitize_value("a" * 5000)) == 1000

    def test_drops_none_and_empty_keys(self) -> None:
        clean = sanitize_metadata({"a": 1, "b": None, "": "x", "c": "  "})
        assert clean == {"a": 1}

    def test_realistic_pdf_metadata_is_fully_scalar(self) -> None:
        clean = sanitize_metadata(
            {
                "page_number": 3,
                "languages": ["eng"],
                "coordinates": {"points": [[0, 0], [1, 1]], "system": "Pixel"},
                "filename": "paper.pdf",
                "parent_id": None,
            }
        )
        assert all(isinstance(v, str | int | float | bool) for v in clean.values())
        assert "parent_id" not in clean


class TestElementStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> ElementStore:
        return ElementStore(tmp_path / "docstore")

    def test_round_trips_an_element(self, store: ElementStore, indexed_items) -> None:
        store.upsert_document(document_id="doc1", filename="a.pdf", pages=2)
        store.put_elements(indexed_items)

        fetched = store.get_elements([item.element_id for item in indexed_items])
        assert len(fetched) == 2

        element, summary = fetched["doc1:figure:0"]
        assert element.kind is ElementKind.FIGURE
        assert element.caption == "Fig. 1. Monthly PM2.5 averages."
        assert element.page_number == 2
        assert "line chart" in summary

    def test_has_document(self, store: ElementStore) -> None:
        assert not store.has_document("doc1")
        store.upsert_document(document_id="doc1", filename="a.pdf", pages=1)
        assert store.has_document("doc1")

    def test_put_elements_is_idempotent(self, store: ElementStore, indexed_items) -> None:
        store.upsert_document(document_id="doc1", filename="a.pdf", pages=2)
        store.put_elements(indexed_items)
        store.put_elements(indexed_items)
        assert store.count_elements("doc1") == 2

    def test_delete_cascades(self, store: ElementStore, indexed_items) -> None:
        store.upsert_document(document_id="doc1", filename="a.pdf", pages=2)
        store.put_elements(indexed_items)
        assert store.delete_document("doc1") == 2
        assert store.count_elements() == 0
        assert not store.has_document("doc1")

    def test_lists_documents_with_counts(self, store: ElementStore, indexed_items) -> None:
        store.upsert_document(document_id="doc1", filename="a.pdf", pages=2)
        store.put_elements(indexed_items)
        records = store.list_documents()
        assert len(records) == 1
        assert records[0].element_count == 2
        assert records[0].filename == "a.pdf"

    def test_get_elements_with_no_ids(self, store: ElementStore) -> None:
        assert store.get_elements([]) == {}

    def test_finds_an_element_by_caption_reference(
        self, store: ElementStore, indexed_items
    ) -> None:
        store.upsert_document(document_id="doc1", filename="a.pdf", pages=2)
        store.put_elements(indexed_items)
        # The fixture figure's caption is "Fig. 1. Monthly PM2.5 averages."
        assert store.find_by_caption_reference({"fig:1"}) == ["doc1:figure:0"]

    def test_reference_lookup_rejects_a_near_miss(self, store: ElementStore, indexed_items) -> None:
        store.upsert_document(document_id="doc1", filename="a.pdf", pages=2)
        store.put_elements(indexed_items)
        # "%1%" matches "Fig. 1." in SQL, so the Python verification must reject it.
        assert store.find_by_caption_reference({"fig:12"}) == []

    def test_reference_lookup_honours_the_document_filter(
        self, store: ElementStore, indexed_items
    ) -> None:
        store.upsert_document(document_id="doc1", filename="a.pdf", pages=2)
        store.put_elements(indexed_items)
        assert store.find_by_caption_reference({"fig:1"}, document_ids=["other"]) == []

    def test_reference_lookup_with_no_references(self, store: ElementStore) -> None:
        assert store.find_by_caption_reference(set()) == []

    def test_survives_reopening(self, tmp_path: Path, indexed_items) -> None:
        directory = tmp_path / "docstore"
        first = ElementStore(directory)
        first.upsert_document(document_id="doc1", filename="a.pdf", pages=2)
        first.put_elements(indexed_items)
        first.close()

        second = ElementStore(directory)
        assert second.count_elements("doc1") == 2


@pytest.mark.usefixtures("stub_embeddings")
class TestMultiVectorIndex:
    @pytest.fixture
    def index(self, settings: Settings) -> MultiVectorIndex:
        return MultiVectorIndex(settings)

    def test_add_and_search(self, index: MultiVectorIndex, indexed_items) -> None:
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        assert index.add_elements(indexed_items) == 2

        results = index.search("monthly PM2.5 chart", k=2)
        assert results
        assert all(source.document_id == "doc1" for source in results)

    def test_search_returns_the_image_path_for_figures(
        self, index: MultiVectorIndex, indexed_items
    ) -> None:
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        index.add_elements(indexed_items)

        figures = index.search("chart", k=5, kinds=[ElementKind.FIGURE])
        assert len(figures) == 1
        assert figures[0].has_image

    def test_kind_filter(self, index: MultiVectorIndex, indexed_items) -> None:
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        index.add_elements(indexed_items)

        texts = index.search("anything", k=5, kinds=[ElementKind.TEXT])
        assert all(source.kind is ElementKind.TEXT for source in texts)

    def test_document_filter_excludes_others(self, index: MultiVectorIndex, indexed_items) -> None:
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        index.add_elements(indexed_items)
        assert index.search("chart", k=5, document_ids=["other"]) == []

    def test_reindexing_upserts_rather_than_duplicating(
        self, index: MultiVectorIndex, indexed_items
    ) -> None:
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        index.add_elements(indexed_items)
        index.add_elements(indexed_items)
        assert index.count_elements("doc1") == 2

    def test_delete_removes_from_both_stores(self, index: MultiVectorIndex, indexed_items) -> None:
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        index.add_elements(indexed_items)
        index.delete_document("doc1")

        assert index.count_elements() == 0
        assert index.search("chart", k=5) == []

    def test_adding_nothing_is_safe(self, index: MultiVectorIndex) -> None:
        assert index.add_elements([]) == 0

    def test_elements_without_text_are_skipped(self, index: MultiVectorIndex) -> None:
        empty = Element(
            element_id="doc1:text:9",
            document_id="doc1",
            kind=ElementKind.TEXT,
            text="",
        )
        index.register_document(document_id="doc1", filename="a.pdf", pages=1, backend="pymupdf")
        assert index.add_elements([IndexedElement(element=empty, summary="")]) == 0

    def test_promotes_an_explicitly_named_figure(
        self, index: MultiVectorIndex, indexed_items
    ) -> None:
        """A question naming "Figure 1" must surface it even if dense search missed."""
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        index.add_elements(indexed_items)

        results = index.search("What does Figure 1 show?", k=1)
        assert results[0].element_id == "doc1:figure:0"

    def test_promotion_fires_when_dense_search_misses(
        self, index: MultiVectorIndex, indexed_items, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case promotion exists for: the named element is outside the top-k."""
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        index.add_elements(indexed_items)

        # Force dense search to return only the text element, as it would on a
        # real paper where a dozen figure captions crowd out the right one.
        hits = index.vectors.similarity_search_with_relevance_scores("chart", k=5)
        only_text = [
            (document, score) for document, score in hits if document.metadata.get("kind") == "text"
        ]
        monkeypatch.setattr(
            index.vectors,
            "similarity_search_with_relevance_scores",
            lambda *args, **kwargs: only_text,
        )

        results = index.search("What does Figure 1 show?", k=3)
        assert results[0].element_id == "doc1:figure:0"
        # Matched by reference rather than similarity, so it carries no score.
        assert results[0].score is None

    def test_no_promotion_without_a_reference(self, index: MultiVectorIndex, indexed_items) -> None:
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        index.add_elements(indexed_items)
        results = index.search("general question about air quality", k=2)
        assert all(source.score is not None for source in results)

    def test_resolve_image(self, index: MultiVectorIndex, indexed_items) -> None:
        index.register_document(document_id="doc1", filename="a.pdf", pages=2, backend="pymupdf")
        index.add_elements(indexed_items)

        assert index.resolve_image("doc1:figure:0") is not None
        assert index.resolve_image("doc1:text:0") is None
        assert index.resolve_image("nonexistent") is None
