"""Extraction backend tests.

These run against a real (generated) PDF rather than mocks, because the value
of this layer is entirely in how it handles actual page geometry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multimodal_rag.config import Settings
from multimodal_rag.domain import BoundingBox, ElementKind, compute_document_id
from multimodal_rag.exceptions import UnsupportedFileError
from multimodal_rag.extractors import get_extractor
from multimodal_rag.extractors.pymupdf_backend import PyMuPDFExtractor
from multimodal_rag.extractors.regions import (
    Region,
    TextBlock,
    attach_captions,
    classify_caption,
    cluster_boxes,
    detect_column_split,
    filter_regions,
    find_caption_anchored_tables,
)


@pytest.fixture
def result(sample_pdf: Path, settings: Settings):
    extractor = PyMuPDFExtractor(settings)
    document_id = compute_document_id(sample_pdf)
    return extractor.extract(sample_pdf, document_id, settings.images_dir / document_id)


class TestPyMuPDFExtractor:
    def test_reports_page_count(self, result) -> None:
        assert result.pages == 2
        assert result.backend == "pymupdf"

    def test_finds_all_three_modalities(self, result) -> None:
        kinds = {element.kind for element in result.elements}
        assert kinds == {ElementKind.TEXT, ElementKind.TABLE, ElementKind.FIGURE}

    def test_renders_an_image_for_every_visual_element(self, result) -> None:
        visuals = [element for element in result.elements if element.kind.is_visual]
        assert visuals, "expected at least one figure or table"
        for element in visuals:
            assert element.has_image, f"{element.element_id} has no rendered crop"
            assert element.image_path.stat().st_size > 0

    def test_binds_captions_to_visuals(self, result) -> None:
        captions = {
            element.kind: element.caption for element in result.elements if element.kind.is_visual
        }
        assert captions[ElementKind.FIGURE].startswith("Fig. 1.")
        assert captions[ElementKind.TABLE].startswith("Table 1.")

    def test_detects_a_table_without_ruling_lines(self, result) -> None:
        # The fixture's table is plain positioned text, which line-based
        # detectors miss entirely; it must be recovered from its caption.
        tables = [e for e in result.elements if e.kind is ElementKind.TABLE]
        assert len(tables) == 1
        assert tables[0].page_number == 2

    def test_text_chunks_carry_page_numbers(self, result) -> None:
        texts = [element for element in result.elements if element.kind is ElementKind.TEXT]
        assert texts
        assert all(element.page_number in (1, 2) for element in texts)

    def test_captions_are_not_duplicated_into_prose(self, result) -> None:
        prose = " ".join(
            element.text for element in result.elements if element.kind is ElementKind.TEXT
        )
        assert "Fig. 1." not in prose

    def test_element_ids_are_unique(self, result) -> None:
        ids = [element.element_id for element in result.elements]
        assert len(ids) == len(set(ids))

    def test_rejects_a_missing_file(self, settings: Settings, tmp_path: Path) -> None:
        extractor = PyMuPDFExtractor(settings)
        with pytest.raises(UnsupportedFileError):
            extractor.extract(tmp_path / "nope.pdf", "x", tmp_path)

    def test_rejects_a_file_that_is_not_a_pdf(self, settings: Settings, tmp_path: Path) -> None:
        fake = tmp_path / "fake.pdf"
        fake.write_text("this is not a pdf")
        extractor = PyMuPDFExtractor(settings)
        with pytest.raises(UnsupportedFileError):
            extractor.extract(fake, "x", tmp_path / "images")


class TestGetExtractor:
    def test_returns_the_configured_backend(self, settings: Settings) -> None:
        assert get_extractor(settings).name == "pymupdf"

    def test_override_wins_over_settings(self, settings: Settings) -> None:
        assert get_extractor(settings, "unstructured").name == "unstructured"


class TestCaptionClassification:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Fig. 1. A chart", ElementKind.FIGURE),
            ("Figure 12: results", ElementKind.FIGURE),
            ("Table 2. Error metrics", ElementKind.TABLE),
            ("Table A.5 Comparison of models", ElementKind.TABLE),
            ("The table below shows", None),
            ("Figures are common", None),
            ("", None),
        ],
    )
    def test_classify(self, text: str, expected: ElementKind | None) -> None:
        assert classify_caption(text) is expected


class TestClustering:
    def test_merges_nearby_boxes(self) -> None:
        boxes = [BoundingBox(0, 0, 10, 10), BoundingBox(12, 0, 22, 10)]
        clusters = cluster_boxes(boxes, gap=5)
        assert len(clusters) == 1
        assert clusters[0].bbox.x1 == 22

    def test_keeps_distant_boxes_apart(self) -> None:
        boxes = [BoundingBox(0, 0, 10, 10), BoundingBox(200, 200, 210, 210)]
        assert len(cluster_boxes(boxes, gap=5)) == 2

    def test_empty_input(self) -> None:
        assert cluster_boxes([], gap=5) == []

    def test_filter_drops_tiny_and_page_sized_regions(self) -> None:
        page_area = 600 * 800
        regions = [
            Region(bbox=BoundingBox(0, 0, 20, 20), has_raster=True),  # too small
            Region(bbox=BoundingBox(0, 0, 599, 799), has_raster=True),  # whole page
            Region(bbox=BoundingBox(50, 50, 350, 350), has_raster=True),  # keep
        ]
        kept = filter_regions(
            regions, page_area=page_area, min_area_ratio=0.012, max_area_ratio=0.95
        )
        assert len(kept) == 1
        assert kept[0].bbox.x0 == 50

    def test_filter_rejects_sparse_vector_clusters(self) -> None:
        # A lone rectangle is a box border, not a chart.
        regions = [Region(bbox=BoundingBox(50, 50, 350, 350), has_raster=False, primitive_count=1)]
        kept = filter_regions(
            regions, page_area=600 * 800, min_area_ratio=0.012, max_area_ratio=0.95
        )
        assert kept == []


class TestCaptionBinding:
    def test_caption_below_a_region_is_attached(self) -> None:
        regions = [Region(bbox=BoundingBox(100, 100, 400, 300))]
        blocks = [
            TextBlock(bbox=BoundingBox(100, 310, 400, 322), text="Table 3. Results by model"),
            TextBlock(bbox=BoundingBox(100, 500, 400, 560), text="Unrelated prose."),
        ]
        attach_captions(regions, blocks)
        assert regions[0].caption.startswith("Table 3.")
        # The caption reclassifies the region even though it looked like a figure.
        assert regions[0].kind is ElementKind.TABLE

    def test_far_away_caption_is_ignored(self) -> None:
        regions = [Region(bbox=BoundingBox(100, 100, 400, 300))]
        blocks = [TextBlock(bbox=BoundingBox(100, 700, 400, 712), text="Fig. 9. Elsewhere")]
        attach_captions(regions, blocks)
        assert regions[0].caption is None

    def test_side_by_side_caption_is_ignored(self) -> None:
        regions = [Region(bbox=BoundingBox(100, 100, 300, 300))]
        blocks = [TextBlock(bbox=BoundingBox(400, 150, 550, 162), text="Fig. 4. Beside it")]
        attach_captions(regions, blocks)
        assert regions[0].caption is None


class TestCaptionAnchoredTables:
    def test_builds_a_region_from_caption_and_rows(self) -> None:
        page = BoundingBox(0, 0, 600, 800)
        blocks = [
            TextBlock(bbox=BoundingBox(50, 100, 550, 112), text="Table 1. Model comparison"),
            TextBlock(bbox=BoundingBox(50, 118, 550, 130), text="Model RMSE MAE"),
            TextBlock(bbox=BoundingBox(50, 134, 550, 146), text="MLR 35.5 21.6"),
            TextBlock(bbox=BoundingBox(50, 150, 550, 162), text="GRU 33.8 22.1"),
            # A large gap ends the table.
            TextBlock(bbox=BoundingBox(50, 400, 550, 460), text="Following paragraph."),
        ]
        regions = find_caption_anchored_tables(blocks, page_bbox=page)
        assert len(regions) == 1
        assert regions[0].kind is ElementKind.TABLE
        assert regions[0].bbox.y1 < 300, "table must not swallow the next paragraph"

    def test_no_caption_means_no_region(self) -> None:
        blocks = [TextBlock(bbox=BoundingBox(50, 100, 550, 160), text="Just prose here.")]
        assert find_caption_anchored_tables(blocks, page_bbox=BoundingBox(0, 0, 600, 800)) == []


class TestColumnDetection:
    def test_detects_two_columns(self) -> None:
        page = BoundingBox(0, 0, 600, 800)
        blocks = [
            TextBlock(bbox=BoundingBox(40, y, 280, y + 40), text="left") for y in range(0, 400, 50)
        ]
        blocks += [
            TextBlock(bbox=BoundingBox(320, y, 560, y + 40), text="right")
            for y in range(0, 400, 50)
        ]
        assert detect_column_split(blocks, page) == pytest.approx(300.0)

    def test_single_column_returns_none(self) -> None:
        page = BoundingBox(0, 0, 600, 800)
        blocks = [
            TextBlock(bbox=BoundingBox(40, y, 560, y + 40), text="wide") for y in range(0, 500, 50)
        ]
        assert detect_column_split(blocks, page) is None
