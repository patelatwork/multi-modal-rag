"""Prompt templates.

Kept in one module so prompt changes are reviewable in isolation -- they are
the highest-leverage, least-tested part of a RAG system.
"""

from __future__ import annotations

from multimodal_rag.domain import ElementKind

FIGURE_SUMMARY_PROMPT = """\
You are indexing a figure extracted from a PDF so it can be found by semantic search later.

Describe the image in a single dense paragraph. Include:
- what kind of visual it is (line chart, bar chart, map, schematic, photograph, ...)
- the subject matter and any title
- axis names, units, series names and legend entries
- the direction of any trend, and the specific values at notable points
- any conclusion the visual is clearly making

Write plain prose with no preamble, no headings, and no markdown formatting.
Never speculate about content you cannot see. If the image is illegible, reply exactly: UNREADABLE
"""

TABLE_SUMMARY_PROMPT = """\
You are indexing a table extracted from a PDF.

Reply with exactly two sections and nothing else:

TABLE:
<the table transcribed as a GitHub-flavoured Markdown table, preserving every row,
column, header and value exactly as shown; use an empty cell for blanks>

SUMMARY:
<one paragraph naming the table's subject, its columns, and the notable
comparisons, extremes or totals a reader would care about>

Transcribe only what is visible. Never invent rows or values.
If the image contains no table, reply exactly: NO_TABLE
"""

TEXT_SUMMARY_PROMPT = """\
Summarise the following passage for a retrieval index. Keep it compact, faithful and
information dense; preserve named entities, numbers and units. Reply with the summary only.

PASSAGE:
{content}
"""

ANSWER_SYSTEM_PROMPT = """\
You are a research assistant answering questions about a set of PDF documents.

You are given retrieved excerpts: prose passages, transcribed tables, and figure images.
Ground every claim in that evidence.

Rules:
- Answer only from the provided context. If it does not contain the answer, say so plainly
  and state what is missing. Do not fall back on general knowledge.
- Inspect any images directly; they are pages, charts and tables from the source documents.
- Cite evidence inline as [S1], [S2], ... matching the source numbers given below.
- When the answer involves several values, present them as a Markdown table.
- Reproduce numbers exactly as they appear. Never estimate a figure the context does not state.
- Use Markdown. Be concise and lead with the answer.
"""

_SUMMARY_PROMPTS = {
    ElementKind.FIGURE: FIGURE_SUMMARY_PROMPT,
    ElementKind.TABLE: TABLE_SUMMARY_PROMPT,
}


def summary_prompt_for(kind: ElementKind) -> str:
    """Return the indexing prompt used for a visual element kind."""
    return _SUMMARY_PROMPTS.get(kind, FIGURE_SUMMARY_PROMPT)


def build_context_block(index: int, kind: ElementKind, page: int | None, body: str) -> str:
    """Render one retrieved source as a labelled block the model can cite."""
    location = f", page {page}" if page is not None else ""
    return f"[S{index}] ({kind.value}{location})\n{body}".strip()
