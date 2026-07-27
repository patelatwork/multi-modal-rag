"""Command line interface.

``python -m multimodal_rag.cli --help``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from multimodal_rag.config import get_settings
from multimodal_rag.exceptions import MultimodalRagError
from multimodal_rag.logging_config import configure_logging, get_logger
from multimodal_rag.pipeline.ingest import ingest_directory, ingest_pdf
from multimodal_rag.pipeline.query import answer_question
from multimodal_rag.storage.vector_store import get_index

logger = get_logger(__name__)


def _print_answer(answer) -> None:
    print(f"\n{answer.text}\n")
    if answer.sources:
        print("Sources:")
        for position, source in enumerate(answer.sources, start=1):
            page = f"p{source.page_number}" if source.page_number else "-"
            marker = " [image]" if source.has_image else ""
            score = f"{source.score:.3f}" if source.score is not None else "n/a"
            print(f"  [S{position}] {source.kind.value:<7} {page:<5} score={score}{marker}")
    print(
        f"\n({len(answer.sources)} sources, {answer.images_sent_to_model} images sent, "
        f"{answer.elapsed_seconds:.1f}s)"
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    target = Path(args.path)

    def progress(done: int, total: int) -> None:
        print(f"\r  summarising {done}/{total}", end="", flush=True)

    if target.is_dir():
        reports = ingest_directory(target, settings=settings, force=args.force)
    else:
        reports = [ingest_pdf(target, settings=settings, force=args.force, on_progress=progress)]
        print()

    for report in reports:
        if report.warnings and not report.document_id:
            print(f"  FAILED  {report.filename}: {report.warnings[0]}")
        elif report.skipped_existing:
            print(f"  SKIPPED {report.filename} (already indexed as {report.document_id})")
        else:
            print(
                f"  OK      {report.filename}: {report.pages} pages, "
                f"{report.text_chunks} text, {report.tables} tables, "
                f"{report.figures} figures in {report.elapsed_seconds:.1f}s"
            )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    index = get_index(settings)

    if args.question:
        _print_answer(answer_question(args.question, settings=settings, index=index, k=args.k))
        return 0

    print("Ask a question, or press enter to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        try:
            _print_answer(answer_question(question, settings=settings, index=index, k=args.k))
        except MultimodalRagError as error:
            print(f"Error: {error}", file=sys.stderr)


def cmd_documents(args: argparse.Namespace) -> int:
    records = get_index().list_documents()
    if not records:
        print("No documents indexed yet.")
        return 0
    print(f"{'DOCUMENT ID':<18} {'PAGES':>5} {'ELEMENTS':>9}  FILENAME")
    for record in records:
        print(
            f"{record.document_id:<18} {record.pages:>5} {record.element_count:>9}  "
            f"{record.filename}"
        )
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    index = get_index()
    if not index.has_document(args.document_id):
        print(f"No such document: {args.document_id}", file=sys.stderr)
        return 1
    removed = index.delete_document(args.document_id)
    print(f"Deleted {args.document_id} ({removed} elements)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "multimodal_rag.api.app:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from multimodal_rag.models.embeddings import embedding_dimension
    from multimodal_rag.models.llm import get_chat_model

    settings = get_settings()
    index = get_index(settings)
    ok = True

    print(f"chat model         {settings.groq_model}")
    print(f"embeddings         {settings.embedding_provider.value}:{settings.embedding_model}")
    print(f"extractor          {settings.extractor_backend.value}")
    print(f"vector store       {settings.vector_store_dir}")

    try:
        print(f"embedding dim      {embedding_dimension(settings)}")
    except MultimodalRagError as error:
        ok = False
        print(f"embeddings         FAILED: {error}")

    try:
        get_chat_model(settings).health_check()
        print("chat api           reachable")
    except MultimodalRagError as error:
        ok = False
        print(f"chat api           FAILED: {error}")

    print(f"documents          {len(index.list_documents())}")
    print(f"elements           {index.count_elements()}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimodal-rag",
        description="Multimodal RAG over PDFs: text, tables and figures.",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index a PDF file or a directory of PDFs")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--force", action="store_true", help="Re-index even if already present")
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", help="Ask a question (interactive if none given)")
    ask.add_argument("question", nargs="?", help="Question to answer")
    ask.add_argument("-k", type=int, default=None, help="Number of sources to retrieve")
    ask.set_defaults(func=cmd_ask)

    documents = subparsers.add_parser("documents", help="List indexed documents")
    documents.set_defaults(func=cmd_documents)

    delete = subparsers.add_parser("delete", help="Remove a document from the index")
    delete.add_argument("document_id")
    delete.set_defaults(func=cmd_delete)

    serve = subparsers.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    health = subparsers.add_parser("health", help="Check credentials and connectivity")
    health.set_defaults(func=cmd_health)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level or get_settings().log_level)

    try:
        return args.func(args)
    except MultimodalRagError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
