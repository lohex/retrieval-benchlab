"""Unit tests for corpus-wide dense embedding transformations."""

from __future__ import annotations

import unittest

import numpy as np

from src.embedding_transforms import (
    CalibrationStatistics,
    EmbeddingTransformConfig,
    EmbeddingTransformType,
    transform_embeddings,
)


class EmbeddingTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queries = np.array([[3.0, 6.0]], dtype=np.float32)
        self.documents = np.array([[1.0, 2.0], [5.0, 10.0]], dtype=np.float32)
        self.calibration = CalibrationStatistics(
            mean=(2.0, 4.0),
            std=(2.0, 4.0),
            source_id="calibration:test",
        )

    def test_identity_returns_values_unchanged(self) -> None:
        queries, documents = transform_embeddings(
            self.queries,
            self.documents,
            EmbeddingTransformConfig(),
        )
        np.testing.assert_array_equal(queries, self.queries)
        np.testing.assert_array_equal(documents, self.documents)

    def test_mean_centering_uses_dimension_wise_document_mean(self) -> None:
        config = EmbeddingTransformConfig(
            transform_type=EmbeddingTransformType.MEAN_CENTER,
            calibration=self.calibration,
        )
        queries, documents = transform_embeddings(self.queries, self.documents, config)
        np.testing.assert_allclose(queries, [[1.0, 2.0]])
        np.testing.assert_allclose(documents, [[-1.0, -2.0], [3.0, 6.0]])

    def test_variance_normalization_scales_without_centering(self) -> None:
        config = EmbeddingTransformConfig(
            transform_type=EmbeddingTransformType.VARIANCE_NORMALIZE,
            calibration=self.calibration,
        )
        queries, documents = transform_embeddings(self.queries, self.documents, config)
        np.testing.assert_allclose(queries, [[1.5, 1.5]])
        np.testing.assert_allclose(documents, [[0.5, 0.5], [2.5, 2.5]])

    def test_z_normalization_centers_and_scales(self) -> None:
        config = EmbeddingTransformConfig(
            transform_type=EmbeddingTransformType.Z_NORMALIZE,
            calibration=self.calibration,
        )
        queries, documents = transform_embeddings(self.queries, self.documents, config)
        np.testing.assert_allclose(queries, [[0.5, 0.5]])
        np.testing.assert_allclose(documents, [[-0.5, -0.5], [1.5, 1.5]])

    def test_epsilon_floors_zero_standard_deviation(self) -> None:
        calibration = CalibrationStatistics(
            mean=(0.0, 0.0),
            std=(0.0, 2.0),
            source_id="calibration:zero-std",
        )
        config = EmbeddingTransformConfig(
            transform_type=EmbeddingTransformType.VARIANCE_NORMALIZE,
            calibration=calibration,
            epsilon=0.5,
        )
        queries, _ = transform_embeddings(self.queries, self.documents, config)
        np.testing.assert_allclose(queries, [[6.0, 3.0]])

    def test_non_identity_transform_requires_calibration(self) -> None:
        with self.assertRaises(ValueError):
            EmbeddingTransformConfig(
                transform_type=EmbeddingTransformType.MEAN_CENTER,
            )


if __name__ == "__main__":
    unittest.main()
