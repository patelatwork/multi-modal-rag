"""SQLite-backed element store.

This is the "document" half of the multi-vector retrieval pattern: the vector
store holds a *summary* of each element for semantic matching, while the full
element -- original text, caption, page, and the path to its rendered image --
lives here and is hydrated after the search returns.

SQLite rather than a directory of pickled blobs because ingestion needs to be
idempotent and documents need to be listable and deletable. Those are queries,
and a key-value file store answers none of them without a full scan.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from multimodal_rag.domain import BoundingBox, Element, ElementKind, IndexedElement
from multimodal_rag.exceptions import StorageError
from multimodal_rag.logging_config import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id           TEXT PRIMARY KEY,
    filename              TEXT NOT NULL,
    pages                 INTEGER NOT NULL DEFAULT 0,
    backend               TEXT,
    embedding_fingerprint TEXT,
    stored_pdf_path       TEXT,
    created_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS elements (
    element_id  TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    page_number INTEGER,
    image_path  TEXT,
    caption     TEXT,
    bbox        TEXT,
    source_pdf  TEXT,
    extra       TEXT
);

CREATE INDEX IF NOT EXISTS idx_elements_document ON elements(document_id);
CREATE INDEX IF NOT EXISTS idx_elements_kind ON elements(kind);
"""


@dataclass(slots=True)
class DocumentRecord:
    """Bookkeeping for one ingested PDF."""

    document_id: str
    filename: str
    pages: int
    backend: str | None
    embedding_fingerprint: str | None
    stored_pdf_path: str | None
    created_at: str
    element_count: int = 0


class ElementStore:
    """Thread-safe SQLite store for extracted elements and their summaries."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / "elements.sqlite3"
        # FastAPI runs sync endpoints on a threadpool, so connections are shared
        # across threads; the lock serialises writes.
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        with self._transaction() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._connection as connection:
            yield connection

    # ----------------------------------------------------------------- writing
    def upsert_document(
        self,
        *,
        document_id: str,
        filename: str,
        pages: int,
        backend: str | None = None,
        embedding_fingerprint: str | None = None,
        stored_pdf_path: str | None = None,
    ) -> None:
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id, filename, pages, backend,
                        embedding_fingerprint, stored_pdf_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        filename = excluded.filename,
                        pages = excluded.pages,
                        backend = excluded.backend,
                        embedding_fingerprint = excluded.embedding_fingerprint,
                        stored_pdf_path = excluded.stored_pdf_path
                    """,
                    (
                        document_id,
                        filename,
                        pages,
                        backend,
                        embedding_fingerprint,
                        stored_pdf_path,
                        datetime.now(UTC).isoformat(timespec="seconds"),
                    ),
                )
        except sqlite3.Error as error:
            msg = f"Could not record document {document_id}: {error}"
            raise StorageError(msg) from error

    def put_elements(self, items: Iterable[IndexedElement]) -> int:
        rows = [
            (
                item.element.element_id,
                item.element.document_id,
                item.element.kind.value,
                item.element.text,
                item.summary,
                item.element.page_number,
                str(item.element.image_path) if item.element.image_path else None,
                item.element.caption,
                json.dumps(item.element.bbox.to_dict()) if item.element.bbox else None,
                item.element.source_pdf,
                json.dumps(item.element.extra) if item.element.extra else None,
            )
            for item in items
        ]
        if not rows:
            return 0

        try:
            with self._transaction() as connection:
                connection.executemany(
                    """
                    INSERT INTO elements (
                        element_id, document_id, kind, text, summary,
                        page_number, image_path, caption, bbox, source_pdf, extra
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(element_id) DO UPDATE SET
                        text = excluded.text,
                        summary = excluded.summary,
                        image_path = excluded.image_path,
                        caption = excluded.caption
                    """,
                    rows,
                )
        except sqlite3.Error as error:
            msg = f"Could not store elements: {error}"
            raise StorageError(msg) from error
        return len(rows)

    def delete_document(self, document_id: str) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM elements WHERE document_id = ?", (document_id,)
            )
            removed = cursor.rowcount
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return removed

    # ----------------------------------------------------------------- reading
    def has_document(self, document_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return row is not None

    def get_elements(self, element_ids: list[str]) -> dict[str, tuple[Element, str]]:
        """Fetch elements by id, preserving nothing about order."""
        if not element_ids:
            return {}
        placeholders = ",".join("?" * len(element_ids))
        rows = self._connection.execute(
            f"SELECT * FROM elements WHERE element_id IN ({placeholders})",
            element_ids,
        ).fetchall()
        return {row["element_id"]: (_row_to_element(row), row["summary"]) for row in rows}

    def find_by_caption_reference(
        self, references: set[str], document_ids: list[str] | None = None
    ) -> list[str]:
        """Element ids whose caption names one of ``references`` (e.g. ``fig:13``).

        Exists because re-ranking cannot rescue a candidate dense search never
        returned: "Figure 13" is semantically indistinguishable from "Figure 10",
        so the right element may sit well outside the top-k. This looks it up
        exactly instead.

        Captions are pre-filtered in SQL on the reference number, then verified
        in Python. That scan is cheap at document scale; a corpus in the millions
        would want the parsed references stored in their own indexed column.
        """
        from multimodal_rag.storage.ranking import extract_references

        if not references:
            return []

        clauses = ["caption IS NOT NULL"]
        params: list[object] = []

        numbers = {reference.split(":", 1)[1] for reference in references}
        clauses.append("(" + " OR ".join(["caption LIKE ?"] * len(numbers)) + ")")
        params.extend(f"%{number}%" for number in numbers)

        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            clauses.append(f"document_id IN ({placeholders})")
            params.extend(document_ids)

        query = f"SELECT element_id, caption FROM elements WHERE {' AND '.join(clauses)}"  # noqa: S608
        rows = self._connection.execute(query, params).fetchall()

        return [
            row["element_id"]
            for row in rows
            if extract_references(row["caption"]) & references
        ]

    def list_documents(self) -> list[DocumentRecord]:
        rows = self._connection.execute(
            """
            SELECT d.*, COUNT(e.element_id) AS element_count
            FROM documents d
            LEFT JOIN elements e ON e.document_id = d.document_id
            GROUP BY d.document_id
            ORDER BY d.created_at DESC
            """
        ).fetchall()
        return [
            DocumentRecord(
                document_id=row["document_id"],
                filename=row["filename"],
                pages=row["pages"],
                backend=row["backend"],
                embedding_fingerprint=row["embedding_fingerprint"],
                stored_pdf_path=row["stored_pdf_path"],
                created_at=row["created_at"],
                element_count=row["element_count"],
            )
            for row in rows
        ]

    def count_elements(self, document_id: str | None = None) -> int:
        if document_id:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n FROM elements WHERE document_id = ?", (document_id,)
            ).fetchone()
        else:
            row = self._connection.execute("SELECT COUNT(*) AS n FROM elements").fetchone()
        return int(row["n"])

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _row_to_element(row: sqlite3.Row) -> Element:
    bbox = None
    if row["bbox"]:
        try:
            values = json.loads(row["bbox"])
            bbox = BoundingBox(values["x0"], values["y0"], values["x1"], values["y1"])
        except (ValueError, KeyError, TypeError):
            bbox = None

    extra: dict = {}
    if row["extra"]:
        try:
            extra = json.loads(row["extra"])
        except ValueError:
            extra = {}

    return Element(
        element_id=row["element_id"],
        document_id=row["document_id"],
        kind=ElementKind(row["kind"]),
        text=row["text"] or "",
        page_number=row["page_number"],
        image_path=Path(row["image_path"]) if row["image_path"] else None,
        caption=row["caption"],
        bbox=bbox,
        source_pdf=row["source_pdf"],
        extra=extra,
    )
