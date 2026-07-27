"""Convenience entry point.

Equivalent to ``python -m multimodal_rag.cli``.
"""

from __future__ import annotations

from multimodal_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
