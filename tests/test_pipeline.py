"""Pipeline tests: summarisation, prompt assembly and ingestion guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from multimodal_rag.config import Settings
from multimodal_rag.domain import Element, ElementKind, RetrievedSource, compute_document_id
from multimodal_rag.exceptions import LLMError, UnsupportedFileError
from multimodal_rag.pipeline.ingest import validate_pdf
from multimodal_rag.pipeline.query import build_prompt
from multimodal_rag.pipeline.summarize import (
    split_table_response,
    summarize_element,
    summarize_elements,
)


class TestTableResponseParsing:
    def test_splits_transcription_from_summary(self) -> None:
        table, summary = split_table_response(
            "TABLE:\n| a | b |\n|---|---|\n| 1 | 2 |\n\nSUMMARY:\nTwo columns of numbers."
        )
        assert table.startswith("| a | b |")
        assert summary == "Two columns of numbers."

    def test_unstructured_response_becomes_the_summary(self) -> None:
        table, summary = split_table_response("Just a description, no sections.")
        assert table is None
        assert summary == "Just a description, no sections."

    def test_handles_an_empty_table_section(self) -> None:
        table, summary = split_table_response("TABLE:\n\nSUMMARY:\nNothing to transcribe.")
        assert table is None
        assert summary == "Nothing to transcribe."


class TestSummarizeElement:
    def test_text_is_indexed_verbatim_by_default(
        self, settings: Settings, text_element: Element, fake_model
    ) -> None:
        result = summarize_element(text_element, model=fake_model, settings=settings)
        assert result.summary == text_element.text
        assert fake_model.calls == [], "text must not cost an LLM call by default"

    def test_text_is_summarised_when_enabled(
        self, settings: Settings, text_element: Element, fake_model
    ) -> None:
        settings.summarize_text_chunks = True
        result = summarize_element(text_element, model=fake_model, settings=settings)
        assert result.summary == "stubbed summary"
        assert len(fake_model.calls) == 1

    def test_figure_is_sent_to_the_vision_model(
        self, settings: Settings, figure_element: Element, fake_model
    ) -> None:
        result = summarize_element(figure_element, model=fake_model, settings=settings)
        assert result.summary == "stubbed summary"
        _, images = fake_model.calls[0]
        assert images == [figure_element.image_path]

    def test_table_transcription_is_kept_alongside_the_summary(
        self, settings: Settings, figure_element: Element
    ) -> None:
        from tests.conftest import FakeChatModel

        figure_element.kind = ElementKind.TABLE
        model = FakeChatModel("TABLE:\n| a |\n|---|\n| 1 |\n\nSUMMARY:\nOne column.")
        result = summarize_element(figure_element, model=model, settings=settings)

        assert figure_element.extra["table_markdown"].startswith("| a |")
        # Both halves are embedded so either can match a query.
        assert "One column." in result.summary
        assert "| a |" in result.summary

    def test_unreadable_sentinel_falls_back_to_the_caption(
        self, settings: Settings, figure_element: Element
    ) -> None:
        from tests.conftest import FakeChatModel

        result = summarize_element(
            figure_element, model=FakeChatModel("UNREADABLE"), settings=settings
        )
        assert result.summary == figure_element.caption

    def test_visual_without_an_image_is_not_sent_to_the_model(
        self, settings: Settings, fake_model
    ) -> None:
        element = Element(
            element_id="d:figure:1",
            document_id="d",
            kind=ElementKind.FIGURE,
            text="fallback text",
        )
        result = summarize_element(element, model=fake_model, settings=settings)
        assert result.summary == "fallback text"
        assert fake_model.calls == []


class TestSummarizeElements:
    def test_one_failure_does_not_abandon_the_batch(
        self, settings: Settings, text_element: Element, figure_element: Element
    ) -> None:
        class FlakyModel:
            def complete(self, prompt, *, images=(), system=None):
                raise LLMError("provider exploded")

        settings.summarize_concurrency = 1
        results = summarize_elements(
            [text_element, figure_element], settings=settings, model=FlakyModel()
        )
        # The text element is unaffected; the figure degrades to its caption.
        assert len(results) == 2
        assert any(item.summary == figure_element.caption for item in results)

    def test_progress_is_reported(
        self, settings: Settings, text_element: Element, fake_model
    ) -> None:
        seen: list[tuple[int, int]] = []
        summarize_elements(
            [text_element],
            settings=settings,
            model=fake_model,
            on_progress=lambda d, t: seen.append((d, t)),
        )
        assert seen == [(1, 1)]

    def test_empty_input(self, settings: Settings) -> None:
        assert summarize_elements([], settings=settings) == []


class TestBuildPrompt:
    def _source(self, position: int, kind: ElementKind, image: Path | None = None):
        return RetrievedSource(
            element_id=f"e{position}",
            document_id="doc1",
            kind=kind,
            text=f"content {position}",
            page_number=position,
            image_path=image,
        )

    def test_numbers_sources_for_citation(self, settings: Settings) -> None:
        sources = [self._source(1, ElementKind.TEXT), self._source(2, ElementKind.TEXT)]
        prompt, images = build_prompt("q?", sources, settings)
        assert "[S1]" in prompt and "[S2]" in prompt
        assert images == []

    def test_attaches_visual_sources(self, settings: Settings, figure_element: Element) -> None:
        sources = [
            self._source(1, ElementKind.TEXT),
            self._source(2, ElementKind.FIGURE, figure_element.image_path),
        ]
        _, images = build_prompt("q?", sources, settings)
        assert images == [figure_element.image_path]

    def test_respects_the_image_budget(self, settings: Settings, figure_element: Element) -> None:
        settings.max_images_per_answer = 1
        sources = [
            self._source(i, ElementKind.FIGURE, figure_element.image_path) for i in range(1, 5)
        ]
        _, images = build_prompt("q?", sources, settings)
        assert len(images) == 1

    def test_truncates_when_over_the_character_budget(self, settings: Settings) -> None:
        settings.max_context_chars = 100
        sources = [
            RetrievedSource(
                element_id=f"e{i}",
                document_id="doc1",
                kind=ElementKind.TEXT,
                text="x" * 500,
            )
            for i in range(5)
        ]
        prompt, _ = build_prompt("q?", sources, settings)
        assert "..." in prompt
        # Only the first, most relevant source should survive the budget.
        # Count context blocks only, not the "[S1], [S2], ..." in the trailer.
        context = prompt.split("QUESTION:")[0]
        assert context.count("[S") == 1

    def test_includes_the_question(self, settings: Settings) -> None:
        prompt, _ = build_prompt("What is the RMSE?", [self._source(1, ElementKind.TEXT)], settings)
        assert "What is the RMSE?" in prompt


class TestValidatePdf:
    def test_accepts_a_real_pdf(self, sample_pdf: Path) -> None:
        validate_pdf(sample_pdf)  # must not raise

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedFileError, match="not found"):
            validate_pdf(tmp_path / "absent.pdf")

    def test_rejects_a_wrong_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        with pytest.raises(UnsupportedFileError, match="Only PDF"):
            validate_pdf(path)

    def test_rejects_a_renamed_non_pdf(self, tmp_path: Path) -> None:
        # A .zip renamed to .pdf must not reach the parser.
        path = tmp_path / "sneaky.pdf"
        path.write_bytes(b"PK\x03\x04 this is a zip")
        with pytest.raises(UnsupportedFileError, match="bad file header"):
            validate_pdf(path)


class TestDocumentId:
    def test_is_content_addressed(self, sample_pdf: Path, tmp_path: Path) -> None:
        copy = tmp_path / "renamed.pdf"
        copy.write_bytes(sample_pdf.read_bytes())
        # Same bytes under a different name must map to the same id, so
        # re-uploading skips a full re-ingest.
        assert compute_document_id(sample_pdf) == compute_document_id(copy)

    def test_differs_for_different_content(self, sample_pdf: Path, tmp_path: Path) -> None:
        other = tmp_path / "other.pdf"
        other.write_bytes(sample_pdf.read_bytes() + b"%extra")
        assert compute_document_id(sample_pdf) != compute_document_id(other)
