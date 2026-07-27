"""Metadata coercion for Chroma.

Chroma only accepts ``str``, ``int``, ``float`` and ``bool`` metadata values and
raises on anything else. PDF parsers happily emit lists (``languages``), nested
dicts (``coordinates``) and ``None``, so metadata has to be flattened before it
reaches the collection -- this was a guaranteed crash on ingest before.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

ScalarValue = str | int | float | bool

# Long free text in metadata bloats the index without helping filtering.
_MAX_VALUE_LENGTH = 1000


def sanitize_value(value: Any) -> ScalarValue | None:
    """Coerce one value to something Chroma will store, or ``None`` to drop it."""
    if value is None:
        return None
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        text = value.strip()
        return text[:_MAX_VALUE_LENGTH] if text else None
    if isinstance(value, Mapping | list | tuple | set):
        try:
            encoded = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            encoded = str(value)
        return encoded[:_MAX_VALUE_LENGTH] or None
    return str(value)[:_MAX_VALUE_LENGTH] or None


def sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, ScalarValue]:
    """Return a Chroma-safe copy of ``metadata``, dropping empty entries."""
    clean: dict[str, ScalarValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            continue
        coerced = sanitize_value(value)
        if coerced is not None:
            clean[key] = coerced
    return clean
