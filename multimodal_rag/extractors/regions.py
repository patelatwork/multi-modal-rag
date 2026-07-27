"""Page-geometry heuristics shared by the extraction backends.

The hard part of multimodal PDF parsing is not reading bytes, it is deciding
*which rectangle on the page is a figure*. A chart is usually several raster
images and dozens of vector primitives that only look like one object to a
human. These helpers cluster those primitives back into regions and attach the
caption that names them.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from multimodal_rag.domain import BoundingBox, ElementKind

# "Fig. 3.", "Figure 12:", "Table 2.", "Table A.5" -- the caption forms that
# appear in practice across academic and report PDFs.
CAPTION_PATTERN = re.compile(
    r"^\s*(?P<label>fig(?:ure)?|table|chart|exhibit|plate)\s*\.?\s*(?P<number>[A-Z]?\.?\d+(?:\.\d+)*)\s*[.:\)-]?\s",
    re.IGNORECASE,
)

_TABLE_LABELS = {"table", "exhibit"}

# A caption sits within this many points of the region it names.
CAPTION_SEARCH_DISTANCE = 72.0

# Rules, underlines and hairlines are this thin; they are never figures.
_MIN_PRIMITIVE_SPAN = 3.0

# A standalone region must be at least this large (points) to be worth rendering.
_MIN_REGION_WIDTH = 60.0
_MIN_REGION_HEIGHT = 40.0


@dataclass(slots=True)
class TextBlock:
    """A positioned run of text on a page."""

    bbox: BoundingBox
    text: str

    @property
    def normalized(self) -> str:
        return " ".join(self.text.split())


@dataclass(slots=True)
class Region:
    """A candidate figure or table area on a page."""

    bbox: BoundingBox
    kind: ElementKind = ElementKind.FIGURE
    caption: str | None = None
    primitive_count: int = 1
    has_raster: bool = False


def classify_caption(text: str) -> ElementKind | None:
    """Return the element kind a caption announces, or ``None`` if it isn't one."""
    match = CAPTION_PATTERN.match(text)
    if match is None:
        return None
    label = match.group("label").lower()
    return ElementKind.TABLE if label in _TABLE_LABELS else ElementKind.FIGURE


def is_caption(text: str) -> bool:
    return classify_caption(text) is not None


def cluster_boxes(
    boxes: list[BoundingBox],
    *,
    gap: float,
    has_raster_flags: list[bool] | None = None,
) -> list[Region]:
    """Merge boxes that touch or nearly touch into single regions.

    Repeatedly unions any two clusters whose boxes come within ``gap`` points of
    each other. Page counts are small (tens of primitives), so the simple
    quadratic sweep is faster in practice than building an index.
    """
    if not boxes:
        return []

    flags = has_raster_flags or [False] * len(boxes)
    clusters: list[Region] = [
        Region(bbox=box, primitive_count=1, has_raster=flag)
        for box, flag in zip(boxes, flags, strict=False)
    ]

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters[i].bbox.intersects(clusters[j].bbox, tolerance=gap):
                    clusters[i] = Region(
                        bbox=clusters[i].bbox.merged(clusters[j].bbox),
                        primitive_count=clusters[i].primitive_count + clusters[j].primitive_count,
                        has_raster=clusters[i].has_raster or clusters[j].has_raster,
                    )
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break

    return clusters


def is_meaningful_primitive(box: BoundingBox) -> bool:
    """Reject hairlines, rules and zero-area artifacts before clustering."""
    return box.width >= _MIN_PRIMITIVE_SPAN and box.height >= _MIN_PRIMITIVE_SPAN


def filter_regions(
    regions: list[Region],
    *,
    page_area: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[Region]:
    """Drop regions that are too small (logos, bullets) or too large (page frames)."""
    kept: list[Region] = []
    for region in regions:
        box = region.bbox
        if box.width < _MIN_REGION_WIDTH or box.height < _MIN_REGION_HEIGHT:
            continue
        if page_area <= 0:
            continue
        ratio = box.area / page_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        # A cluster of pure vector strokes needs some substance to count as a
        # chart; a single stray rectangle usually isn't one.
        if not region.has_raster and region.primitive_count < 3:
            continue
        kept.append(region)
    return kept


def attach_captions(regions: list[Region], blocks: list[TextBlock]) -> None:
    """Bind the nearest caption to each region and let it decide the kind.

    A caption wins over geometry: an area announced by "Table 4." is a table
    even if it was detected as a cluster of vector strokes.
    """
    caption_blocks = [block for block in blocks if is_caption(block.normalized)]
    if not caption_blocks:
        return

    used: set[int] = set()
    for region in regions:
        best_index: int | None = None
        best_distance = CAPTION_SEARCH_DISTANCE

        for index, block in enumerate(caption_blocks):
            if index in used:
                continue
            if not _horizontally_aligned(region.bbox, block.bbox):
                continue
            distance = _vertical_distance(region.bbox, block.bbox)
            if distance < best_distance:
                best_distance = distance
                best_index = index

        if best_index is not None:
            block = caption_blocks[best_index]
            used.add(best_index)
            region.caption = block.normalized
            kind = classify_caption(block.normalized)
            if kind is not None:
                region.kind = kind


def _horizontally_aligned(region: BoundingBox, caption: BoundingBox) -> bool:
    """True when the caption sits under/over the region rather than beside it."""
    overlap = min(region.x1, caption.x1) - max(region.x0, caption.x0)
    narrower = min(region.width, caption.width)
    return narrower > 0 and overlap > 0.35 * narrower


def _vertical_distance(region: BoundingBox, caption: BoundingBox) -> float:
    """Gap between a region and a caption; 0 when they overlap vertically."""
    if caption.y0 >= region.y1:
        return caption.y0 - region.y1
    if caption.y1 <= region.y0:
        return region.y0 - caption.y1
    return 0.0


def find_caption_anchored_tables(
    blocks: list[TextBlock],
    *,
    page_bbox: BoundingBox,
) -> list[Region]:
    """Locate tables that have a caption but no ruling lines.

    Line-based detectors find nothing in most academic papers, whose tables use
    only a few horizontal rules or none at all. Anchoring on the "Table N."
    caption and walking down the contiguous run of text below it recovers them.
    The region is rendered as an image and transcribed by the vision model, so
    the bounds only need to be approximately right.
    """
    regions: list[Region] = []
    ordered = sorted(blocks, key=lambda block: (block.bbox.y0, block.bbox.x0))

    for index, block in enumerate(ordered):
        text = block.normalized
        if classify_caption(text) is not ElementKind.TABLE:
            continue

        following = [
            other
            for other in ordered[index + 1 :]
            if other.bbox.y0 >= block.bbox.y1 - 2.0
            and _horizontally_aligned(block.bbox, other.bbox)
        ]
        if not following:
            continue

        gaps = [following[i + 1].bbox.y0 - following[i].bbox.y1 for i in range(len(following) - 1)]
        # Rows inside a table are packed tighter than the gap to the next
        # paragraph, so a jump well above the median gap ends the table.
        gap_limit = max(2.0 * statistics.median(gaps), 18.0) if gaps else 24.0

        body = [following[0]]
        for i in range(len(following) - 1):
            if following[i + 1].bbox.y0 - following[i].bbox.y1 > gap_limit:
                break
            if is_caption(following[i + 1].normalized):
                break
            body.append(following[i + 1])

        bbox = block.bbox
        for member in body:
            bbox = bbox.merged(member.bbox)

        bbox = BoundingBox(
            max(bbox.x0, page_bbox.x0),
            max(bbox.y0, page_bbox.y0),
            min(bbox.x1, page_bbox.x1),
            min(bbox.y1, page_bbox.y1),
        )
        if bbox.height < _MIN_REGION_HEIGHT or bbox.width < _MIN_REGION_WIDTH:
            continue

        regions.append(
            Region(bbox=bbox, kind=ElementKind.TABLE, caption=text, primitive_count=len(body))
        )

    return regions


def detect_column_split(blocks: list[TextBlock], page_bbox: BoundingBox) -> float | None:
    """Return the x of a two-column gutter, or ``None`` for single-column pages.

    Multi-column pages must be read column-by-column; sorting purely by ``y``
    interleaves the two columns into nonsense. A page is treated as two-column
    when almost nothing crosses the centre line and both sides carry text.
    """
    if len(blocks) < 6:
        return None

    centre = (page_bbox.x0 + page_bbox.x1) / 2.0
    tolerance = page_bbox.width * 0.04

    straddling = left = right = 0
    for block in blocks:
        box = block.bbox
        if box.x0 < centre - tolerance and box.x1 > centre + tolerance:
            straddling += 1
        elif box.x1 <= centre + tolerance:
            left += 1
        else:
            right += 1

    if left < 2 or right < 2:
        return None
    if straddling > 0.2 * len(blocks):
        return None
    return centre


def sort_reading_order(blocks: list[TextBlock], page_bbox: BoundingBox) -> list[TextBlock]:
    """Order blocks the way a human reads the page."""
    centre = detect_column_split(blocks, page_bbox)
    if centre is None:
        return sorted(blocks, key=lambda block: (block.bbox.y0, block.bbox.x0))

    def key(block: TextBlock) -> tuple[int, float, float]:
        # Full-width blocks (titles, wide tables) stay in the left stream so
        # they are not pushed below the whole first column.
        box = block.bbox
        is_right = box.x0 >= centre - page_bbox.width * 0.04
        return (1 if is_right else 0, box.y0, box.x0)

    return sorted(blocks, key=key)
