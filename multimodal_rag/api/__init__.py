"""HTTP interface.

The FastAPI instance deliberately is *not* re-exported here: binding the name
``app`` on the package would shadow the :mod:`multimodal_rag.api.app` submodule.
Import it explicitly instead::

    from multimodal_rag.api.app import app
"""

from __future__ import annotations

__all__: list[str] = []
