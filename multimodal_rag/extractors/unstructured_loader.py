from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unstructured.documents.elements import Image, Table
from unstructured.partition.pdf import partition_pdf


@dataclass(slots=True)
class ExtractedElement:
    doc_id: str
    kind: str
    content: str
    metadata: dict[str, Any]


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if hasattr(metadata, "to_dict"):
        return dict(metadata.to_dict())
    if hasattr(metadata, "__dict__"):
        return {key: value for key, value in vars(metadata).items() if value is not None}
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def extract_pdf_elements(pdf_path: str | Path, output_dir: str | Path | None = None) -> list[ExtractedElement]:
    pdf_path = Path(pdf_path)
    image_output_dir = Path(output_dir) if output_dir else pdf_path.parent / f"{pdf_path.stem}_images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],
        extract_image_block_output_dir=str(image_output_dir),
    )

    extracted: list[ExtractedElement] = []
    for index, element in enumerate(elements):
        metadata = _metadata_to_dict(getattr(element, "metadata", None))
        metadata.setdefault("source_pdf", str(pdf_path))
        metadata.setdefault("page_number", getattr(getattr(element, "metadata", None), "page_number", None))

        if isinstance(element, Table):
            kind = "table"
            content = metadata.get("text_as_html") or getattr(element, "text", "") or ""
        elif isinstance(element, Image):
            kind = "image"
            image_path = metadata.get("image_path")
            if image_path:
                metadata["image_path"] = str(Path(image_path))
            content = getattr(element, "text", "") or metadata.get("image_path", "") or ""
        else:
            kind = "text"
            content = getattr(element, "text", "") or ""

        content = str(content).strip()
        if not content and kind != "image":
            continue

        extracted.append(
            ExtractedElement(
                doc_id=f"{pdf_path.stem}-{index}",
                kind=kind,
                content=content,
                metadata=metadata,
            )
        )

    return extracted