"""Turns extracted elements into the text that gets embedded.

Figures and tables have no useful text of their own, so the vision model writes
it for them. That description is the *only* way an image becomes searchable, so
this stage is what makes retrieval multimodal rather than text-only.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from multimodal_rag.config import Settings, get_settings
from multimodal_rag.domain import Element, ElementKind, IndexedElement
from multimodal_rag.exceptions import LLMError
from multimodal_rag.logging_config import get_logger
from multimodal_rag.models.llm import VisionChatModel, get_chat_model
from multimodal_rag.models.prompts import TEXT_SUMMARY_PROMPT, summary_prompt_for

logger = get_logger(__name__)

# Sentinels the prompts instruct the model to return when an image is unusable.
_FAILURE_SENTINELS = {"UNREADABLE", "NO_TABLE"}

_TABLE_SECTION = re.compile(r"^\s*TABLE:\s*(?P<table>.*?)\s*SUMMARY:\s*(?P<summary>.*)$", re.DOTALL)

ProgressCallback = Callable[[int, int], None]


def split_table_response(response: str) -> tuple[str | None, str]:
    """Separate the Markdown transcription from the prose summary.

    Both halves are kept: the transcription is what the UI renders as a real
    table and what lets the answering model quote exact cell values, while the
    prose is what makes the table findable by a natural-language query.
    """
    match = _TABLE_SECTION.match(response.strip())
    if match is None:
        return None, response.strip()
    table = match.group("table").strip()
    summary = match.group("summary").strip()
    return (table or None), summary


def summarize_element(
    element: Element,
    *,
    model: VisionChatModel | None,
    settings: Settings,
) -> IndexedElement:
    """Produce the indexable text for a single element.

    ``model`` may be ``None`` only when nothing in the batch needs one (all
    text, with summarisation disabled).
    """
    if element.kind is ElementKind.TEXT:
        if not settings.summarize_text_chunks or model is None:
            return IndexedElement(element=element, summary=element.text)
        summary = model.complete(TEXT_SUMMARY_PROMPT.format(content=element.text))
        return IndexedElement(element=element, summary=summary or element.text)

    if not element.has_image or model is None:
        # A visual element with no rendered crop is all we can index verbatim.
        return IndexedElement(element=element, summary=element.text)

    response = model.complete(summary_prompt_for(element.kind), images=[element.image_path])
    stripped = response.strip()

    if not stripped or stripped in _FAILURE_SENTINELS:
        logger.warning("Model could not read %s (%s)", element.element_id, stripped or "empty")
        fallback = element.caption or element.text
        return IndexedElement(element=element, summary=fallback)

    if element.kind is ElementKind.TABLE:
        table_markdown, prose = split_table_response(stripped)
        if table_markdown:
            element.extra["table_markdown"] = table_markdown
            # Embedding both means a query can match either the prose
            # description or a specific value inside the table.
            return IndexedElement(element=element, summary=f"{prose}\n\n{table_markdown}")
        return IndexedElement(element=element, summary=prose)

    return IndexedElement(element=element, summary=stripped)


def summarize_elements(
    elements: Sequence[Element],
    *,
    settings: Settings | None = None,
    model: VisionChatModel | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[IndexedElement]:
    """Summarise every element, in parallel, tolerating individual failures.

    One unreadable figure must not abandon a 60-page ingestion, so a failed
    element falls back to its caption and the run continues.
    """
    settings = settings or get_settings()
    elements = list(elements)
    if not elements:
        return []

    needs_model = any(
        element.kind is not ElementKind.TEXT or settings.summarize_text_chunks
        for element in elements
    )
    model = model or (get_chat_model(settings) if needs_model else None)

    total = len(elements)
    completed = 0
    results: list[IndexedElement | None] = [None] * total

    def work(index: int) -> None:
        element = elements[index]
        try:
            results[index] = summarize_element(element, model=model, settings=settings)
        except LLMError as error:
            logger.warning("Summary failed for %s: %s", element.element_id, error)
            results[index] = IndexedElement(
                element=element, summary=element.caption or element.text
            )

    # Visual elements are I/O bound on the API, so threads are the right tool;
    # concurrency stays low because vision calls burn tokens fast.
    with ThreadPoolExecutor(max_workers=settings.summarize_concurrency) as pool:
        for _ in pool.map(work, range(total)):
            completed += 1
            if on_progress:
                on_progress(completed, total)

    return [item for item in results if item is not None and item.summary.strip()]
