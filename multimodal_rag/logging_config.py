"""Logging setup.

Call :func:`configure_logging` exactly once per process (the CLI and the API
both do). Library modules should only ever call :func:`logging.getLogger`.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

# Third-party libraries that are chatty at INFO and drown out our own logs.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "chromadb",
    "sentence_transformers",
    "urllib3",
    "unstructured",
    "pdfminer",
    "PIL",
)


def configure_logging(level: str = "INFO") -> None:
    """Install a single stderr handler and quiet down noisy dependencies."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
