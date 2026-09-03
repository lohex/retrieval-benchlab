"""Tests for retrieval evaluation configuration and persistence."""

from __future__ import annotations

import sqlite3
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


def _install_dependency_stubs() -> None:
    module = types.ModuleType("datasets")
    module.Dataset = object
    module.load_dataset = lambda *args, **kwargs: None
    module.load_from_disk = lambda *args, **kwargs: None
    sys.modules.setdefault("datasets", module)


_install_dependency_stubs()

import src.evaluate as evaluation
import src.evaluation_registry as registry
from src.embedding_transforms import (
    CalibrationStatistics,
    EmbeddingTransformConfig,
    EmbeddingTransformType,
)
from src.evaluation_models import BioASQSample, RuntimeConfig
from src.retrievers import RetrieverType


class _CalibrationModel:
    def __init__(self) -> None:
        self.arguments = {}

    def encode_document(self, documents, **kwargs):
        self.arguments = kwargs
        return np.array([[1.0, 2.0], [3.0, 6.0]])


class _FakeRetriever:
    def rank(self, queries, corpus, *, top_k):
        return {query_id: list(corpus)[:top_k] for query_id in queries}


def _sample() -> BioASQSample:
    return BioASQSample(
        queries={"q1": "query"},
        relevant_docs={"q1": {"d1"}},
        corpus={"d1": "relevant", "d2": "background"},
        metadata={"n_queries": 1, "n_corpus_docs": 2},
    )


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry_path = self.root / "datasets.sqlite"
        self.results_path = self.root / "results.sqlite"
        self.patches = [
            patch.object(evaluation, "mount_google_drive", return_value=None),
            patch.object(registry, "mount_google_drive", return_value=None),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def test_public_api_is_reexported(self) -> None:
        self.assertIs(evaluation.register_pipeline, registry.register_pipeline)
        self.assertIs(evaluation.register_evaluation, registry.register_evaluation)
        self.assertIs(evaluation.register_dataset, registry.register_dataset)

    def test_runtime_settings_do_not_change_pipeline_identity(self) -> None:
        first = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "cosine",
            registry_db_path=self.registry_path,
        )
        second = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "cosine",
            registry_db_path=self.registry_path,
        )
        self.assertEqual(first, second)
        with registry.open_registry_database(self.registry_path) as connection:
            _, config_json = registry.get_pipeline(connection, first)
        self.assertNotIn("batch_size", config_json)
        self.assertNotIn("corpus_scan_size", config_json)

    def test_transform_and_calibration_change_pipeline_identity(self) -> None:
        raw_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "cosine",
            registry_db_path=self.registry_path,
        )
        calibration = CalibrationStatistics(
            mean=(1.0, 2.0),
            std=(0.5, 1.0),
            source_id="calibration:test-v1",
        )
        centered_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "cosine",
            embedding_transform=EmbeddingTransformConfig(
                transform_type=EmbeddingTransformType.MEAN_CENTER,
                calibration=calibration,
            ),
            registry_db_path=self.registry_path,
        )
        self.assertNotEqual(raw_id, centered_id)
        with registry.open_registry_database(self.registry_path) as connection:
            definition, config_json = registry.get_pipeline(connection, centered_id)
        self.assertIs(
            definition.embedding_transform.transform_type,
            EmbeddingTransformType.MEAN_CENTER,
        )
        self.assertIn("calibration:test-v1", config_json)

    def test_bm25_and_dense_are_distinct_pipeline_types(self) -> None:
        dense_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            registry_db_path=self.registry_path,
        )
        bm25_id = evaluation.register_pipeline(
            retriever_type=RetrieverType.BM25,
            registry_db_path=self.registry_path,
        )
        self.assertNotEqual(dense_id, bm25_id)

    def test_compute_calibration_statistics_uses_raw_document_embeddings(self) -> None:
        model = _CalibrationModel()
        statistics = evaluation.compute_calibration_statistics(
            model,
            ["one", "two"],
            source_id="calibration:test",
            batch_size=2,
            show_progress_bar=False,
        )
        self.assertEqual(statistics.mean, (2.0, 4.0))
        self.assertEqual(statistics.std, (1.0, 2.0))
        self.assertFalse(model.arguments["normalize_embeddings"])

    def test_result_identity_ignores_runtime(self) -> None:
        dataset_path = self.root / "datasets" / "list-one"
        dataset_path.mkdir(parents=True)
        (dataset_path / "metadata.json").write_text("{}", encoding="utf-8")
        (dataset_path / "corpus").mkdir()
        pipeline_id = evaluation.register_pipeline(
            retriever_type="bm25",
            registry_db_path=self.registry_path,
        )
        evaluation_id = evaluation.register_evaluation(
            registry_db_path=self.registry_path,
        )
        with (
            patch.object(evaluation, "load_bioasq_sample", return_value=_sample()),
            patch.object(evaluation, "_build_retriever", return_value=_FakeRetriever()),
        ):
            first = evaluation.evaluate(
                pipeline_id,
                dataset_path.parent,
                evaluation_id=evaluation_id,
                runtime=RuntimeConfig(show_progress_bar=False),
                registry_db_path=self.registry_path,
                results_db_path=self.results_path,
            )
            second = evaluation.evaluate(
                pipeline_id,
                dataset_path.parent,
                evaluation_id=evaluation_id,
                runtime=RuntimeConfig(batch_size=1, show_progress_bar=False),
                registry_db_path=self.registry_path,
                results_db_path=self.results_path,
            )
        self.assertEqual(first[0].status, evaluation.EvaluationStatus.EVALUATED)
        self.assertEqual(second[0].status, evaluation.EvaluationStatus.SKIPPED_EXISTING)
        with sqlite3.connect(self.results_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0]
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
