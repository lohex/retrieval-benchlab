"""Tests for query-adapted dense similarity scoring."""

from __future__ import annotations

import unittest

import numpy as np

from src.embedding_transforms import (
    CalibrationStatistics,
    EmbeddingTransformConfig,
    EmbeddingTransformType,
)
from src.retrievers import DenseRetriever, DenseRetrieverConfig


class QueryAdaptedScoringTests(unittest.TestCase):
    def _retriever(self, alpha: float) -> DenseRetriever:
        calibration = CalibrationStatistics(
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            source_id="calibration:test",
        )
        retriever = DenseRetriever.__new__(DenseRetriever)
        retriever.config = DenseRetrieverConfig(
            model_name="test-model",
            similarity_metric="cosine",
            embedding_transform=EmbeddingTransformConfig(
                transform_type=EmbeddingTransformType.QUERY_ADAPTED_Z,
                calibration=calibration,
                alpha=alpha,
            ),
        )
        return retriever

    def test_alpha_zero_matches_standard_cosine(self) -> None:
        queries = np.array([[1.0, 2.0, -1.0]], dtype=np.float32)
        documents = np.array(
            [[2.0, 1.0, 0.0], [-1.0, 1.0, 2.0]],
            dtype=np.float32,
        )
        retriever = self._retriever(alpha=0.0)
        scores = retriever._weighted_cosine_block(queries, documents)
        expected = (
            queries @ documents.T
            / (
                np.linalg.norm(queries, axis=1)[:, None]
                * np.linalg.norm(documents, axis=1)[None, :]
            )
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-6)

    def test_positive_alpha_changes_dimension_weights(self) -> None:
        query = np.array([[4.0, 1.0]], dtype=np.float32)
        documents = np.array([[1.0, 4.0], [4.0, 1.0]], dtype=np.float32)
        retriever = self._retriever(alpha=1.0)
        retriever.config = DenseRetrieverConfig(
            model_name="test-model",
            similarity_metric="cosine",
            embedding_transform=EmbeddingTransformConfig(
                transform_type=EmbeddingTransformType.QUERY_ADAPTED_Z,
                calibration=CalibrationStatistics(
                    mean=(0.0, 0.0),
                    std=(1.0, 1.0),
                    source_id="calibration:test-2d",
                ),
                alpha=1.0,
            ),
        )
        scores = retriever._weighted_cosine_block(query, documents)
        self.assertGreater(float(scores[0, 1]), float(scores[0, 0]))


if __name__ == "__main__":
    unittest.main()
