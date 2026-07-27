"""API contract tests.

The chat model and embeddings are stubbed: these assert routing, validation,
serialisation and error mapping, not answer quality.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from multimodal_rag.config import Settings
from multimodal_rag.domain import Answer, ElementKind, RetrievedSource
from multimodal_rag.exceptions import LLMError
from multimodal_rag.storage.vector_store import MultiVectorIndex


@pytest.fixture
def client(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, stub_embeddings: None
) -> Iterator[TestClient]:
    from multimodal_rag.api import app as app_module

    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "get_index", lambda _=None: MultiVectorIndex(settings))

    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def seeded(client: TestClient, indexed_items) -> TestClient:
    index: MultiVectorIndex = client.app.state.index
    index.register_document(document_id="doc1", filename="paper.pdf", pages=2, backend="pymupdf")
    index.add_elements(indexed_items)
    return client


class TestHealth:
    def test_reports_configuration(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["extractor_backend"] == "pymupdf"
        assert body["embedding_provider"] == "local"

    def test_degrades_without_an_api_key(self, client: TestClient, settings: Settings) -> None:
        settings.groq_api_key = None
        body = client.get("/api/health").json()
        assert body["status"] == "degraded"
        assert "GROQ_API_KEY" in body["detail"]


class TestDocuments:
    def test_empty_index(self, client: TestClient) -> None:
        assert client.get("/api/documents").json() == []

    def test_lists_a_seeded_document(self, seeded: TestClient) -> None:
        documents = seeded.get("/api/documents").json()
        assert len(documents) == 1
        assert documents[0]["filename"] == "paper.pdf"
        assert documents[0]["element_count"] == 2

    def test_delete(self, seeded: TestClient) -> None:
        assert seeded.delete("/api/documents/doc1").status_code == 204
        assert seeded.get("/api/documents").json() == []

    def test_delete_unknown_is_404(self, client: TestClient) -> None:
        assert client.delete("/api/documents/nope").status_code == 404

    def test_rejects_a_non_pdf_upload(self, client: TestClient) -> None:
        response = client.post(
            "/api/documents", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 415

    def test_rejects_an_oversized_upload(self, client: TestClient, settings: Settings) -> None:
        settings.max_upload_mb = 1
        oversized = b"%PDF-" + b"0" * (2 * 1024 * 1024)
        response = client.post(
            "/api/documents", files={"file": ("big.pdf", oversized, "application/pdf")}
        )
        assert response.status_code == 413


class TestChat:
    def test_rejects_an_empty_question(self, client: TestClient) -> None:
        assert client.post("/api/chat", json={"question": ""}).status_code == 422

    def test_rejects_an_out_of_range_k(self, client: TestClient) -> None:
        assert client.post("/api/chat", json={"question": "hi", "k": 99}).status_code == 422

    def test_returns_answer_and_sources(
        self, seeded: TestClient, monkeypatch: pytest.MonkeyPatch, figure_element
    ) -> None:
        from multimodal_rag.api import app as app_module

        def fake_answer(question, **kwargs):
            return Answer(
                question=question,
                text="SA-GNN performs best [S1].",
                sources=[
                    RetrievedSource(
                        element_id="doc1:figure:0",
                        document_id="doc1",
                        kind=ElementKind.FIGURE,
                        text="A chart of monthly averages.",
                        page_number=2,
                        caption="Fig. 1.",
                        image_path=figure_element.image_path,
                        score=0.82,
                    )
                ],
                images_sent_to_model=1,
                elapsed_seconds=1.234,
            )

        monkeypatch.setattr(app_module, "answer_question", fake_answer)
        body = seeded.post("/api/chat", json={"question": "Which is best?"}).json()

        assert body["answer"].startswith("SA-GNN")
        assert body["images_sent_to_model"] == 1
        source = body["sources"][0]
        assert source["label"] == "S1"
        assert source["kind"] == "figure"
        # The UI needs a URL it can drop straight into an <img> tag.
        assert source["image_url"] == "/api/elements/doc1:figure:0/image"

    def test_extracts_table_markdown_for_the_ui(
        self, seeded: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from multimodal_rag.api import app as app_module

        def fake_answer(question, **kwargs):
            return Answer(
                question=question,
                text="See the table.",
                sources=[
                    RetrievedSource(
                        element_id="doc1:table:0",
                        document_id="doc1",
                        kind=ElementKind.TABLE,
                        text="Error metrics by model.\n\n| Model | RMSE |\n|---|---|\n| MLR | 35.5 |",
                        page_number=3,
                    )
                ],
            )

        monkeypatch.setattr(app_module, "answer_question", fake_answer)
        body = seeded.post("/api/chat", json={"question": "table?"}).json()
        assert body["sources"][0]["table_markdown"].startswith("| Model | RMSE |")

    def test_llm_failure_maps_to_502(
        self, seeded: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from multimodal_rag.api import app as app_module

        def failing(question, **kwargs):
            raise LLMError("provider unavailable")

        monkeypatch.setattr(app_module, "answer_question", failing)
        response = seeded.post("/api/chat", json={"question": "hi"})
        assert response.status_code == 502
        assert response.json()["kind"] == "LLMError"


class TestElementImage:
    def test_serves_a_rendered_crop(self, seeded: TestClient) -> None:
        response = seeded.get("/api/elements/doc1:figure:0/image")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 0

    def test_404_for_a_text_element(self, seeded: TestClient) -> None:
        assert seeded.get("/api/elements/doc1:text:0/image").status_code == 404

    def test_404_for_an_unknown_element(self, seeded: TestClient) -> None:
        assert seeded.get("/api/elements/nope/image").status_code == 404

    def test_refuses_a_path_outside_the_image_root(
        self, seeded: TestClient, tmp_path: Path
    ) -> None:
        """A poisoned image_path must not turn into arbitrary file read."""
        secret = tmp_path / "secret.txt"
        secret.write_text("classified")

        index: MultiVectorIndex = seeded.app.state.index
        with index.store._transaction() as connection:
            connection.execute(
                "UPDATE elements SET image_path = ? WHERE element_id = ?",
                (str(secret), "doc1:figure:0"),
            )

        assert seeded.get("/api/elements/doc1:figure:0/image").status_code == 404


class TestJobs:
    def test_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get("/api/jobs/does-not-exist").status_code == 404
