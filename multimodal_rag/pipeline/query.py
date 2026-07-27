"""Question answering over the multimodal index.

The step that makes this multimodal rather than a text RAG with pictures
attached: retrieved figures and tables are sent to the model **as images**, so
it reads the chart itself instead of relying on the description written at
ingest time. The same sources come back with the answer, which is how the UI
can show the chart the answer is talking about.
"""

from __future__ import annotations

import time
from pathlib import Path

from multimodal_rag.config import Settings, get_settings
from multimodal_rag.domain import Answer, ElementKind, RetrievedSource
from multimodal_rag.logging_config import get_logger
from multimodal_rag.models.llm import VisionChatModel, get_chat_model
from multimodal_rag.models.prompts import ANSWER_SYSTEM_PROMPT, build_context_block
from multimodal_rag.storage.vector_store import MultiVectorIndex, get_index

logger = get_logger(__name__)

_NO_CONTEXT_MESSAGE = (
    "I could not find anything relevant in the indexed documents. "
    "Try rephrasing the question, or ingest a document that covers it."
)


def retrieve(
    question: str,
    *,
    settings: Settings | None = None,
    index: MultiVectorIndex | None = None,
    k: int | None = None,
    document_ids: list[str] | None = None,
    kinds: list[ElementKind] | None = None,
) -> list[RetrievedSource]:
    """Search the index for evidence relevant to ``question``."""
    settings = settings or get_settings()
    index = index or get_index(settings)
    return index.search(question, k=k, document_ids=document_ids, kinds=kinds)


def build_prompt(
    question: str, sources: list[RetrievedSource], settings: Settings
) -> tuple[str, list[Path]]:
    """Assemble the text context and pick which images to attach.

    Images are budgeted rather than sent wholesale: they dominate token cost,
    and beyond a handful they crowd out the text evidence.
    """
    blocks: list[str] = []
    images: list[Path] = []
    image_labels: list[str] = []
    budget = settings.max_context_chars

    for position, source in enumerate(sources, start=1):
        body = source.text.strip()
        if source.caption and source.caption not in body:
            body = f"{source.caption}\n{body}".strip()

        if len(body) > budget:
            body = body[:budget].rstrip() + " ..."
        budget -= len(body)

        attach = (
            source.has_image
            and source.kind.is_visual
            and len(images) < settings.max_images_per_answer
        )
        if attach:
            images.append(Path(source.image_path))  # type: ignore[arg-type]
            image_labels.append(f"S{position}")
            body = f"{body}\n(The image for this source is attached below.)".strip()

        blocks.append(build_context_block(position, source.kind, source.page_number, body))

        if budget <= 0:
            logger.debug("Context budget exhausted after %s sources", position)
            break

    context = "\n\n".join(blocks)
    attachment_note = (
        f"\n\nAttached images, in order: {', '.join(image_labels)}." if image_labels else ""
    )
    prompt = (
        f"CONTEXT:\n{context}{attachment_note}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above, citing sources as [S1], [S2], ..."
    )
    return prompt, images


def answer_question(
    question: str,
    *,
    settings: Settings | None = None,
    index: MultiVectorIndex | None = None,
    model: VisionChatModel | None = None,
    k: int | None = None,
    document_ids: list[str] | None = None,
) -> Answer:
    """Retrieve evidence and synthesise a grounded, cited answer."""
    settings = settings or get_settings()
    index = index or get_index(settings)
    started = time.perf_counter()

    question = question.strip()
    if not question:
        return Answer(question=question, text="Please ask a question.", sources=[])

    sources = retrieve(question, settings=settings, index=index, k=k, document_ids=document_ids)
    if not sources:
        return Answer(
            question=question,
            text=_NO_CONTEXT_MESSAGE,
            sources=[],
            elapsed_seconds=time.perf_counter() - started,
        )

    prompt, images = build_prompt(question, sources, settings)
    model = model or get_chat_model(settings)

    logger.info(
        "Answering %r with %s sources (%s images attached)",
        question[:60],
        len(sources),
        len(images),
    )
    text = model.complete(prompt, images=images, system=ANSWER_SYSTEM_PROMPT)

    return Answer(
        question=question,
        text=text,
        sources=sources,
        images_sent_to_model=len(images),
        elapsed_seconds=time.perf_counter() - started,
    )
