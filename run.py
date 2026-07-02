from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_rag.config import get_settings
from multimodal_rag.extractors.unstructured_loader import extract_pdf_elements
from multimodal_rag.models.llm import answer_question, summarize_image, summarize_table, summarize_text
from multimodal_rag.storage.vector_store import build_indexed_items, ingest_items


def summarize_element(kind: str, content: str, metadata: dict[str, object]) -> str:
    if kind == "table":
        return summarize_table(content)
    if kind == "image":
        image_path = metadata.get("image_path")
        if image_path:
            return summarize_image(image_path)
        return content
    return summarize_text(content)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal chat-to-PDF pipeline")
    parser.add_argument("pdf", type=Path, help="Path to the PDF to ingest")
    parser.add_argument("--question", type=str, help="Ask a question after indexing")
    return parser


def main() -> None:
    settings = get_settings()
    args = build_arg_parser().parse_args()

    extracted_elements = extract_pdf_elements(args.pdf, settings.extraction_dir)
    summaries: dict[str, str] = {}

    for element in extracted_elements:
        summaries[element.doc_id] = summarize_element(element.kind, element.content, element.metadata)

    indexed_items = build_indexed_items(extracted_elements, summaries)
    retriever = ingest_items(indexed_items, settings)

    if args.question:
        documents = retriever.invoke(args.question)
        print(answer_question(args.question, documents))
        return

    while True:
        question = input("Question (enter to quit): ").strip()
        if not question:
            break
        documents = retriever.invoke(question)
        print(answer_question(question, documents))


if __name__ == "__main__":
    main()