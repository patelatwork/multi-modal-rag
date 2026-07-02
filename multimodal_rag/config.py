from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    input_pdf_dir: Path
    extraction_dir: Path
    vector_store_dir: Path
    docstore_dir: Path
    collection_name: str
    llm_model: str
    llm_api_key: str | None
    llm_temperature: float
    hf_embedding_model: str
    hf_api_key: str | None
    hf_api_url: str | None
    retrieval_k: int

    def ensure_directories(self) -> None:
        for directory in (
            self.input_pdf_dir,
            self.extraction_dir,
            self.vector_store_dir,
            self.docstore_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    project_root = Path(__file__).resolve().parents[1]
    llm_api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
    vector_store_dir = os.getenv("VECTOR_STORE_DIR") or os.getenv("DB_DIR")
    extraction_dir = os.getenv("EXTRACTION_DIR") or os.getenv("EXTRACTED_IMAGES_DIR")
    settings = AppConfig(
        project_root=project_root,
        input_pdf_dir=Path(os.getenv("INPUT_PDF_DIR", project_root / "data" / "pdfs")),
        extraction_dir=Path(extraction_dir or project_root / "data" / "extracted"),
        vector_store_dir=Path(vector_store_dir or project_root / "storage" / "chroma"),
        docstore_dir=Path(os.getenv("DOCSTORE_DIR", project_root / "storage" / "docstore")),
        collection_name=os.getenv("CHROMA_COLLECTION", "multimodal_rag"),
        llm_model=os.getenv("GROQ_MODEL") or os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        llm_api_key=llm_api_key,
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        hf_embedding_model=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        hf_api_key=os.getenv("HF_API_TOKEN"),
        hf_api_url=os.getenv("HF_API_URL"),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
    )
    settings.ensure_directories()
    return settings