"""Tests for backend-independent retrieval metrics."""

from __future__ import annotations

import unittest

from src.metrics import evaluate_rankings


class RetrievalMetricTests(unittest.TestCase):
    def test_perfect_ranking_scores_one(self) -> None:
        rankings = {"q1": ["d1", "d2", "d3"]}
        relevant = {"q1": {"d1", "d2"}}
        metrics = evaluate_rankings(
            rankings,
            relevant,
            {
                "mrr_at_k": (10,),
                "ndcg_at_k": (10,),
                "accuracy_at_k": (1,),
                "precision_recall_at_k": (1, 2),
                "map_at_k": (10,),
            },
        )
        self.assertEqual(metrics["mrr@10"], 1.0)
        self.assertEqual(metrics["ndcg@10"], 1.0)
        self.assertEqual(metrics["accuracy@1"], 1.0)
        self.assertEqual(metrics["precision@2"], 1.0)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["map@10"], 1.0)

    def test_query_ids_must_match(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_rankings(
                {"q1": ["d1"]},
                {"q2": {"d1"}},
                {"mrr_at_k": (1,)},
            )


if __name__ == "__main__":
    unittest.main()
