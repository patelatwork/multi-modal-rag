"""Hybrid re-ranking.

The motivating failure: asking "what does Figure 13 show" over a paper with
thirteen figures returns Figures 5, 7 and 10, because a sentence encoder sees
almost no difference between their captions.
"""

from __future__ import annotations

import pytest

from multimodal_rag.storage.ranking import (
    Candidate,
    extract_references,
    hybrid_rank,
    lexical_scores,
    tokenize,
)


class TestTokenize:
    def test_drops_stopwords_and_punctuation(self) -> None:
        assert tokenize("What is the RMSE of the model?") == ["rmse", "model"]

    def test_keeps_decimal_identifiers_intact(self) -> None:
        assert "a.5" in tokenize("See Table A.5 for details")

    def test_empty_input(self) -> None:
        assert tokenize("") == []


class TestExtractReferences:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Figure 13 shows", {"fig:13"}),
            ("Fig. 13. Critical difference diagram", {"fig:13"}),
            ("see figure13", {"fig:13"}),
            ("Table A.5 reports", {"table:a.5"}),
            ("Both Fig. 2 and Table 3", {"fig:2", "table:3"}),
            ("no references here", set()),
        ],
    )
    def test_extraction(self, text: str, expected: set[str]) -> None:
        assert extract_references(text) == expected

    def test_spelling_variants_normalise_together(self) -> None:
        # "Figure 13" in a question must match "Fig. 13." in a caption.
        assert extract_references("Figure 13") == extract_references("Fig. 13.")


class TestLexicalScores:
    def test_exact_reference_outranks_a_sibling(self) -> None:
        candidates = [
            Candidate("a", "Fig. 10. Residual plot for the predicted data.", 0.9),
            Candidate("b", "Fig. 13. Model-wise critical difference diagram.", 0.9),
        ]
        scores = lexical_scores("What does Figure 13 show?", candidates)
        assert scores[1] > scores[0]

    def test_returns_zero_without_query_signal(self) -> None:
        candidates = [Candidate("a", "some text", 0.5)]
        assert lexical_scores("the of and", candidates) == [0.0]

    def test_rare_terms_beat_common_ones(self) -> None:
        candidates = [
            Candidate("a", "the model was evaluated on the model data", 0.5),
            Candidate("b", "heteroscedasticity was observed in the model", 0.5),
        ]
        scores = lexical_scores("heteroscedasticity", candidates)
        assert scores[1] > scores[0]


class TestHybridRank:
    def test_lexical_signal_rescues_the_right_figure(self) -> None:
        """The exact scenario that motivated this module."""
        candidates = [
            Candidate("f10", "Fig. 10. Residual plot for PM2.5-GNN.", 0.92),
            Candidate("f5", "Fig. 5. Input data and model architecture.", 0.91),
            Candidate("f13", "Fig. 13. Model-wise critical difference diagram.", 0.88),
        ]
        ranked = hybrid_rank("Summarise what Figure 13 shows.", candidates, top_k=1)
        assert ranked[0][0].key == "f13", "the named figure must win"

    def test_dense_order_survives_without_a_reference(self) -> None:
        candidates = [
            Candidate("a", "Air quality varies seasonally across the region.", 0.95),
            Candidate("b", "Completely unrelated boilerplate text.", 0.30),
        ]
        ranked = hybrid_rank("seasonal air quality", candidates, top_k=2)
        assert [candidate.key for candidate, _ in ranked] == ["a", "b"]

    def test_respects_top_k(self) -> None:
        candidates = [Candidate(str(i), f"text {i}", 0.5) for i in range(10)]
        assert len(hybrid_rank("text", candidates, top_k=3)) == 3

    def test_zero_weight_is_pure_dense(self) -> None:
        candidates = [
            Candidate("low", "Figure 13 mentioned here", 0.10),
            Candidate("high", "unrelated", 0.99),
        ]
        ranked = hybrid_rank("Figure 13", candidates, top_k=1, lexical_weight=0.0)
        assert ranked[0][0].key == "high"

    def test_empty_candidates(self) -> None:
        assert hybrid_rank("anything", [], top_k=5) == []
