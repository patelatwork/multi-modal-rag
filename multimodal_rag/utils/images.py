"""Image encoding helpers for multimodal prompts.

Vision APIs bill by pixels and reject oversized payloads, so every image is
downscaled before it is base64-encoded. A 200-DPI crop of a full page is
comfortably over 4 MB raw; the same crop at 1600 px on its longest edge costs a
fraction of that and is still legible to the model.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

from PIL import Image

from multimodal_rag.exceptions import MultimodalRagError
from multimodal_rag.logging_config import get_logger

logger = get_logger(__name__)

_MAX_DIMENSION = 1600
_JPEG_QUALITY = 85


def get_image_mime_type(image_path: str | Path) -> str:
    """Best-effort MIME type for a path, defaulting to PNG."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/png"


def encode_image(image_path: str | Path) -> str:
    """Base64-encode a file verbatim, without re-compressing it."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def load_image_data_url(
    image_path: str | Path,
    *,
    max_dimension: int = _MAX_DIMENSION,
    jpeg_quality: int = _JPEG_QUALITY,
) -> str:
    """Return a ``data:`` URL for ``image_path``, downscaling when oversized.

    Images with transparency stay PNG (flattening them can erase thin chart
    lines drawn on an alpha background); everything else becomes JPEG, which is
    typically 5-10x smaller for the same rendered crop.
    """
    path = Path(image_path)
    if not path.is_file():
        msg = f"Image not found: {path}"
        raise MultimodalRagError(msg)

    try:
        with Image.open(path) as image:
            image.load()
            has_alpha = image.mode in ("RGBA", "LA", "P")
            longest = max(image.size)

            if longest <= max_dimension and has_alpha:
                return f"data:{get_image_mime_type(path)};base64,{encode_image(path)}"

            if longest > max_dimension:
                scale = max_dimension / longest
                new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                image = image.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            if has_alpha:
                image.convert("RGBA").save(buffer, format="PNG", optimize=True)
                mime = "image/png"
            else:
                image.convert("RGB").save(
                    buffer, format="JPEG", quality=jpeg_quality, optimize=True
                )
                mime = "image/jpeg"

            payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:{mime};base64,{payload}"
    except OSError as error:
        msg = f"Could not read image {path}: {error}"
        raise MultimodalRagError(msg) from error


# Kept so existing callers and notebooks keep working.
load_image_base64_data_url = load_image_data_url
