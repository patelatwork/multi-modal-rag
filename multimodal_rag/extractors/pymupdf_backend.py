"""PyMuPDF extraction backend (default).

Chosen as the default because it ships as a self-contained wheel: no poppler,
no tesseract, no system packages. It reads text with layout information and --
crucially for multimodal RAG -- can rasterise an arbitrary rectangle of a page.

That last capability drives the design. Rather than trying to parse table cells
back out of a PDF's drawing commands (brittle, and it mangles most real
tables), we locate the *region* a table or figure occupies, render it to PNG,
and let the vision model read it. The model then produces both a retrieval
summary and a Markdown transcription, and the PNG itself is what the UI shows.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multimodal_rag.config import Settings, get_settings
from multimodal_rag.domain import BoundingBox, Element, ElementKind
from multimodal_rag.exceptions import ExtractionError, UnsupportedFileError
from multimodal_rag.extractors.base import ExtractionResult
from multimodal_rag.extractors.regions import (
    Region,
    TextBlock,
    attach_captions,
    cluster_boxes,
    filter_regions,
    find_caption_anchored_tables,
    is_caption,
    is_meaningful_primitive,
    sort_reading_order,
)
from multimodal_rag.logging_config import get_logger

logger = get_logger(__name__)

# Pad rendered crops so axis labels and table rules are not clipped.
_RENDER_PADDING = 8.0

# Text blocks with this much of their area inside a figure are the figure's own
# labels; they are read by the vision model instead of being indexed as prose.
_ENCLOSURE_THRESHOLD = 0.7


class PyMuPDFExtractor:
    """Extracts text chunks, figures and tables using PyMuPDF."""

    name = "pymupdf"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------ public
    def extract(self, pdf_path: Path, document_id: str, image_dir: Path) -> ExtractionResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            msg = f"PDF not found: {pdf_path}"
            raise UnsupportedFileError(msg)

        image_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []

        try:
            document = pymupdf.open(pdf_path)
        except Exception as error:  # pymupdf raises a variety of low-level errors
            msg = f"Could not open PDF {pdf_path.name}: {error}"
            raise UnsupportedFileError(msg) from error

        try:
            if document.needs_pass:
                msg = f"PDF {pdf_path.name} is password protected"
                raise UnsupportedFileError(msg)

            visual_elements: list[Element] = []
            page_texts: list[tuple[int, str]] = []

            for page_index in range(document.page_count):
                page = document[page_index]
                page_number = page_index + 1
                try:
                    regions, body_text = self._process_page(
                        page=page,
                        page_number=page_number,
                        document_id=document_id,
                        image_dir=image_dir,
                        pdf_path=pdf_path,
                    )
                except Exception as error:  # one bad page must not kill the run
                    logger.warning("Page %s of %s failed: %s", page_number, pdf_path.name, error)
                    warnings.append(f"page {page_number}: {error}")
                    continue

                visual_elements.extend(regions)
                if body_text.strip():
                    page_texts.append((page_number, body_text))

            text_elements = self._chunk_pages(page_texts, document_id, pdf_path)
            elements = text_elements + visual_elements
            pages = document.page_count
        finally:
            document.close()

        if not elements:
            msg = (
                f"No content could be extracted from {pdf_path.name}. "
                "It may be a scanned document; try the unstructured backend with OCR enabled."
            )
            raise ExtractionError(msg)

        logger.info(
            "Extracted %s elements from %s (%s pages) via %s",
            len(elements),
            pdf_path.name,
            pages,
            self.name,
        )
        return ExtractionResult(
            document_id=document_id,
            source_pdf=pdf_path,
            pages=pages,
            elements=elements,
            warnings=warnings,
            backend=self.name,
        )

    # ----------------------------------------------------------------- private
    def _process_page(
        self,
        *,
        page: pymupdf.Page,
        page_number: int,
        document_id: str,
        image_dir: Path,
        pdf_path: Path,
    ) -> tuple[list[Element], str]:
        """Return the page's visual elements and its remaining body text."""
        page_bbox = BoundingBox(*page.rect)
        blocks = self._text_blocks(page)

        regions = self._detect_regions(page, page_bbox, blocks)
        attach_captions(regions, blocks)

        elements: list[Element] = []
        for ordinal, region in enumerate(regions):
            element = self._render_region(
                page=page,
                region=region,
                ordinal=ordinal,
                page_number=page_number,
                document_id=document_id,
                image_dir=image_dir,
                pdf_path=pdf_path,
            )
            if element is not None:
                elements.append(element)

        body_text = self._body_text(blocks, regions, page_bbox)
        return elements, body_text

    @staticmethod
    def _text_blocks(page: pymupdf.Page) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        for raw in page.get_text("blocks"):
            x0, y0, x1, y1, text = raw[0], raw[1], raw[2], raw[3], raw[4]
            if not isinstance(text, str) or not text.strip():
                continue
            blocks.append(TextBlock(bbox=BoundingBox(x0, y0, x1, y1), text=text))
        return blocks

    def _detect_regions(
        self,
        page: pymupdf.Page,
        page_bbox: BoundingBox,
        blocks: list[TextBlock],
    ) -> list[Region]:
        """Find every figure and table area on the page."""
        boxes: list[BoundingBox] = []
        raster_flags: list[bool] = []

        for info in page.get_image_info():
            box = BoundingBox(*info["bbox"])
            if is_meaningful_primitive(box):
                boxes.append(box)
                raster_flags.append(True)

        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is None:
                continue
            box = BoundingBox(rect.x0, rect.y0, rect.x1, rect.y1)
            if is_meaningful_primitive(box):
                boxes.append(box)
                raster_flags.append(False)

        clustered = cluster_boxes(
            boxes, gap=self.settings.region_merge_gap, has_raster_flags=raster_flags
        )
        regions = filter_regions(
            clustered,
            page_area=page_bbox.area,
            min_area_ratio=self.settings.min_region_area_ratio,
            max_area_ratio=self.settings.max_region_area_ratio,
        )

        # Tables that are pure text (no rules, no raster) are invisible to the
        # clustering above, so recover them from their captions.
        for table in find_caption_anchored_tables(blocks, page_bbox=page_bbox):
            if not any(table.bbox.intersects(existing.bbox) for existing in regions):
                regions.append(table)

        regions.sort(key=lambda region: (region.bbox.y0, region.bbox.x0))
        return regions

    def _render_region(
        self,
        *,
        page: pymupdf.Page,
        region: Region,
        ordinal: int,
        page_number: int,
        document_id: str,
        image_dir: Path,
        pdf_path: Path,
    ) -> Element | None:
        """Rasterise one region to PNG and wrap it in an :class:`Element`."""
        clip = pymupdf.Rect(*region.bbox.expanded(_RENDER_PADDING).as_tuple()) & page.rect
        if clip.is_empty or clip.width < 1 or clip.height < 1:
            return None

        pixmap = page.get_pixmap(clip=clip, dpi=self.settings.render_dpi)
        if pixmap.width == 0 or pixmap.height == 0:
            return None

        filename = f"p{page_number:04d}_{ordinal:02d}_{region.kind.value}.png"
        image_path = image_dir / filename
        pixmap.save(image_path)

        # Text drawn inside the region (axis labels, table cells) is kept as a
        # hint for the summariser -- it is often garbled, so the model treats
        # the rendered image as the source of truth.
        raw_text = page.get_textbox(clip).strip()

        return Element(
            element_id=f"{document_id}:p{page_number}:{region.kind.value}:{ordinal}",
            document_id=document_id,
            kind=region.kind,
            text=raw_text,
            page_number=page_number,
            image_path=image_path,
            caption=region.caption,
            bbox=region.bbox,
            source_pdf=str(pdf_path),
            extra={"render_dpi": self.settings.render_dpi},
        )

    @staticmethod
    def _body_text(
        blocks: list[TextBlock],
        regions: list[Region],
        page_bbox: BoundingBox,
    ) -> str:
        """Page prose, with figure internals and captions removed."""
        kept: list[TextBlock] = []
        for block in blocks:
            if is_caption(block.normalized):
                continue  # captions travel with their figure element instead
            if any(_enclosed(block.bbox, region.bbox) for region in regions):
                continue
            kept.append(block)

        ordered = sort_reading_order(kept, page_bbox)
        return "\n\n".join(block.text.strip() for block in ordered if block.text.strip())

    def _chunk_pages(
        self,
        page_texts: list[tuple[int, str]],
        document_id: str,
        pdf_path: Path,
    ) -> list[Element]:
        """Split the document's prose into overlapping chunks.

        Pages are concatenated before splitting so a chunk can span a page
        break, then each chunk is mapped back to the page it starts on via the
        splitter's start index.
        """
        if not page_texts:
            return []

        separator = "\n\n"
        spans: list[tuple[int, int, int]] = []  # (start, end, page_number)
        cursor = 0
        pieces: list[str] = []
        for page_number, text in page_texts:
            pieces.append(text)
            spans.append((cursor, cursor + len(text), page_number))
            cursor += len(text) + len(separator)

        full_text = separator.join(pieces)
        documents = self._splitter.create_documents([full_text])

        elements: list[Element] = []
        for ordinal, document in enumerate(documents):
            content = document.page_content.strip()
            if len(content) < 40:  # page furniture: headers, folios, stray glyphs
                continue
            start = int(document.metadata.get("start_index", 0))
            page_number = _page_for_offset(spans, start)
            elements.append(
                Element(
                    element_id=f"{document_id}:text:{ordinal}",
                    document_id=document_id,
                    kind=ElementKind.TEXT,
                    text=content,
                    page_number=page_number,
                    source_pdf=str(pdf_path),
                )
            )
        return elements


def _enclosed(inner: BoundingBox, outer: BoundingBox) -> bool:
    """True when most of ``inner`` lies inside ``outer``."""
    overlap_width = min(inner.x1, outer.x1) - max(inner.x0, outer.x0)
    overlap_height = min(inner.y1, outer.y1) - max(inner.y0, outer.y0)
    if overlap_width <= 0 or overlap_height <= 0 or inner.area <= 0:
        return False
    return (overlap_width * overlap_height) / inner.area >= _ENCLOSURE_THRESHOLD


def _page_for_offset(spans: list[tuple[int, int, int]], offset: int) -> int | None:
    for start, end, page_number in spans:
        if start <= offset <= end:
            return page_number
    return spans[-1][2] if spans else None
