"""Hybrid re-ranking for retrieval.

Dense embeddings are good at meaning and bad at identifiers. "Figure 13" and
"Figure 10" are nearly identical vectors, so a question naming a specific figure
or table often retrieves the wrong one -- every caption in a paper looks alike to
a sentence encoder.

The fix is to blend the dense score with a lexical one computed over the same
candidates: rare query terms that appear verbatim in an element pull it up, and
an explicit reference like "Table A.5" is treated as a near-exact match. No
extra index and no extra dependency; the vector store is simply over-fetched and
the candidates re-scored.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

# "Figure 13", "Fig. 3", "Table A.5" -- the identifiers dense retrieval fumbles.
REFERENCE_PATTERN = re.compile(
    r"\b(?P<label>fig(?:ure)?|table|chart|exhibit)s?\.?\s*(?P<number>[A-Za-z]?\.?\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")

# Words too common to carry signal. Kept as prose rather than a list literal so the
# vocabulary stays scannable; split through a name so SIM905 does not rewrite it.
_STOPWORD_TEXT = """
a an and are as at be by does do for from has have how in into is it its of on or
that the their there these this to was were what when where which who why will with
show shows showing tell me about give please compare between
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

# Weight of the lexical score in the blend. Dense still leads: lexical is there
# to break ties and rescue exact references, not to run the ranking.
LEXICAL_WEIGHT = 0.35

# A matching "Figure 13" is worth more than any ordinary term overlap.
REFERENCE_BONUS = 1.0


@dataclass(slots=True)
class Candidate:
    """A retrieval hit awaiting re-ranking."""

    key: str
    text: str
    dense_score: float


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


def extract_references(text: str) -> set[str]:
    """Normalised figure/table references found in ``text``.

    "Fig. 13", "Figure 13" and "figure13" all collapse to ``fig:13`` so a
    question phrased either way matches the caption either way.
    """
    references: set[str] = set()
    for match in REFERENCE_PATTERN.finditer(text):
        label = match.group("label").lower()
        label = "fig" if label.startswith("fig") else label
        number = match.group("number").lstrip(".").lower()
        references.add(f"{label}:{number}")
    return references


def _inverse_document_frequency(documents: Sequence[list[str]]) -> dict[str, float]:
    """Rare terms score higher, so shared boilerplate does not dominate."""
    total = len(documents) or 1
    frequencies: Counter[str] = Counter()
    for tokens in documents:
        frequencies.update(set(tokens))
    return {term: math.log(1.0 + total / (1.0 + count)) for term, count in frequencies.items()}


def _normalize(scores: list[float]) -> list[float]:
    """Scale to 0-1 so the two score families are comparable."""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-9:
        return [1.0 if high > 0 else 0.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def lexical_scores(query: str, candidates: Sequence[Candidate]) -> list[float]:
    """Score each candidate on verbatim overlap with the query."""
    query_tokens = set(tokenize(query))
    query_references = extract_references(query)
    if not query_tokens and not query_references:
        return [0.0] * len(candidates)

    tokenized = [tokenize(candidate.text) for candidate in candidates]
    idf = _inverse_document_frequency(tokenized)

    raw: list[float] = []
    for candidate, tokens in zip(candidates, tokenized, strict=True):
        present = set(tokens)
        overlap = sum(idf.get(term, 0.0) for term in query_tokens & present)
        # Longer elements accumulate overlap by luck; damp that.
        score = overlap / math.sqrt(1 + len(present))

        if query_references and query_references & extract_references(candidate.text):
            score += REFERENCE_BONUS

        raw.append(score)

    return _normalize(raw)


def hybrid_rank(
    query: str,
    candidates: Sequence[Candidate],
    *,
    top_k: int,
    lexical_weight: float = LEXICAL_WEIGHT,
) -> list[tuple[Candidate, float]]:
    """Blend dense and lexical scores, returning the best ``top_k`` candidates."""
    if not candidates:
        return []

    # Dense scores are used as-is, only clamped. Min-max normalising them would
    # stretch a meaningless 0.88-vs-0.92 gap across the full range and let it
    # outweigh an exact "Figure 13" match. Lexical scores are unbounded raw
    # sums, so those genuinely do need scaling.
    dense = [min(max(candidate.dense_score, 0.0), 1.0) for candidate in candidates]
    lexical = lexical_scores(query, candidates)

    ranked = [
        (candidate, (1.0 - lexical_weight) * dense_score + lexical_weight * lexical_score)
        for candidate, dense_score, lexical_score in zip(candidates, dense, lexical, strict=True)
    ]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
