"""Generates a synthetic PDF exercising every extraction path.

Tests must not depend on a copyrighted paper sitting in ``data/``, so the
fixture builds its own document containing the three shapes the extractor has
to handle: two-column prose, a raster chart with a ``Fig.`` caption, and a
rule-less table with a ``Table`` caption.
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf

_LOREM = (
    "Urban air quality modelling combines meteorological reanalysis with ground "
    "sensor observations to forecast particulate concentrations. The proposed "
    "architecture couples a spatial attention module with a temporal encoder so "
    "that station level readings inform one another across the monitoring "
    "network. Ablation results isolate the contribution of each component. "
)

_TABLE_ROWS = [
    ("Model", "RMSE", "MAE", "R2"),
    ("MLR", "35.57", "21.62", "0.80"),
    ("LSTM", "39.66", "26.41", "0.75"),
    ("GRU", "33.87", "22.14", "0.82"),
    ("SA-GNN", "29.33", "16.20", "0.86"),
]


def _chart_png() -> bytes:
    """A simple bar chart drawn with PyMuPDF, returned as PNG bytes."""
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=260)
    page.draw_line(pymupdf.Point(50, 210), pymupdf.Point(390, 210), width=1.5)
    page.draw_line(pymupdf.Point(50, 210), pymupdf.Point(50, 30), width=1.5)

    values = [0.80, 0.75, 0.82, 0.86]
    labels = ["MLR", "LSTM", "GRU", "SA-GNN"]
    for index, (value, label) in enumerate(zip(values, labels, strict=True)):
        x0 = 75 + index * 78
        height = value * 190
        page.draw_rect(
            pymupdf.Rect(x0, 210 - height, x0 + 46, 210),
            color=(0.15, 0.35, 0.65),
            fill=(0.25, 0.5, 0.8),
        )
        page.insert_text(pymupdf.Point(x0 + 4, 226), label, fontsize=9)
        page.insert_text(pymupdf.Point(x0 + 8, 205 - height), f"{value:.2f}", fontsize=8)

    page.insert_text(pymupdf.Point(150, 22), "R2 score by model", fontsize=11)
    pixmap = page.get_pixmap(dpi=150)
    data = pixmap.tobytes("png")
    doc.close()
    return data


def build_sample_pdf(destination: Path) -> Path:
    """Write the fixture PDF to ``destination`` and return the path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()

    # --- page 1: two-column prose plus a captioned chart -------------------
    page = doc.new_page(width=595, height=842)
    page.insert_text(pymupdf.Point(50, 60), "Multimodal Retrieval Benchmarks", fontsize=16)

    chart = _chart_png()
    chart_rect = pymupdf.Rect(90, 90, 505, 330)
    page.insert_image(chart_rect, stream=io.BytesIO(chart))
    page.insert_textbox(
        pymupdf.Rect(90, 336, 505, 366),
        "Fig. 1. R2 score achieved by each baseline model on the held-out split.",
        fontsize=9,
    )

    left = pymupdf.Rect(50, 390, 285, 780)
    right = pymupdf.Rect(310, 390, 545, 780)
    page.insert_textbox(left, _LOREM * 2, fontsize=9.5, align=pymupdf.TEXT_ALIGN_JUSTIFY)
    page.insert_textbox(right, _LOREM * 2, fontsize=9.5, align=pymupdf.TEXT_ALIGN_JUSTIFY)

    # --- page 2: a rule-less table introduced by its caption ---------------
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        pymupdf.Rect(50, 60, 545, 90),
        "Table 1. Forecast error comparison across candidate architectures.",
        fontsize=9,
    )
    y = 105
    for row_index, row in enumerate(_TABLE_ROWS):
        for column_index, cell in enumerate(row):
            page.insert_text(
                pymupdf.Point(60 + column_index * 110, y),
                cell,
                fontsize=9.5,
                fontname="hebo" if row_index == 0 else "helv",
            )
        y += 17

    page.insert_textbox(
        pymupdf.Rect(50, 250, 545, 700),
        _LOREM * 4,
        fontsize=9.5,
        align=pymupdf.TEXT_ALIGN_JUSTIFY,
    )

    doc.save(destination)
    doc.close()
    return destination


if __name__ == "__main__":  # pragma: no cover - manual fixture regeneration
    target = Path(__file__).resolve().parents[2] / "data" / "pdfs" / "sample_report.pdf"
    print(build_sample_pdf(target))
