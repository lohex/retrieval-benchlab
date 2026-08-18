"""Unit tests for retrieval evaluation configuration and persistence."""

from __future__ import annotations

import json
import sqlite3
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


def _install_dependency_stubs() -> None:
    datasets_module = types.ModuleType("datasets")
    datasets_module.Dataset = object
    datasets_module.load_dataset = lambda *args, **kwargs: None
    datasets_module.load_from_disk = lambda *args, **kwargs: None
    sys.modules.setdefault("datasets", datasets_module)


_install_dependency_stubs()

import src.evaluate as evaluation
import src.evaluation_registry as registry
import src.io as bio_io
from src.evaluation_models import BioASQSample, CalibrationSet, RuntimeConfig
from src.retrievers import RetrieverType


class _CalibrationModel:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    def encode_document(self, documents, **kwargs):
        self.arguments = {"documents": list(documents), **kwargs}
        return np.array([[1.0, 2.0], [3.0, 6.0]])

    def encode(self, documents, **kwargs):
        raise AssertionError("Document calibration must use encode_document")


def _sample_payload(content_marker: str = "v1") -> BioASQSample:
    return BioASQSample(
        queries={"q1": "query"},
        relevant_docs={"q1": {"d1"}},
        corpus={"d1": f"relevant-{content_marker}", "d2": "background"},
        metadata={"subset_name": "sample", "n_queries": 1, "n_corpus_docs": 2},
    )


class _FakeRetriever:
    def rank(self, queries, corpus, *, top_k):
        return {query_id: ["d1", "d2"][:top_k] for query_id in queries}


class EvaluationRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.registry_path = self.root / "datasets.sqlite"
        self.results_path = self.root / "results.sqlite"
        self.drive_patches = [
            patch.object(evaluation, "mount_google_drive", return_value=None),
            patch.object(registry, "mount_google_drive", return_value=None),
        ]
        for drive_patch in self.drive_patches:
            drive_patch.start()

    def tearDown(self) -> None:
        for drive_patch in self.drive_patches:
            drive_patch.stop()
        self.temp_directory.cleanup()

    def _create_dataset_directory(self, name: str) -> Path:
        path = self.root / "datasets" / name
        path.mkdir(parents=True)
        (path / "metadata.json").write_text("{}", encoding="utf-8")
        (path / "corpus").mkdir()
        return path

    def test_public_api_is_reexported(self) -> None:
        self.assertIs(evaluation.register_pipeline, registry.register_pipeline)
        self.assertIs(evaluation.register_evaluation, registry.register_evaluation)
        self.assertIs(evaluation.register_dataset, registry.register_dataset)

    def test_io_loaders_return_typed_models(self) -> None:
        dataset_path = self._create_dataset_directory("list-one")
        metadata = {
            "queries": {"q1": "query"},
            "relevant_docs": {"q1": ["d1"]},
            "n_queries": 1,
            "n_corpus_docs": 1,
        }
        (dataset_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with patch.object(
            bio_io,
            "load_from_disk",
            return_value={"doc_id": ["d1"], "text": ["document"]},
        ):
            sample = bio_io.load_bioasq_sample(dataset_path)
        self.assertIsInstance(sample, BioASQSample)

        calibration_path = self.root / "calibration" / "bioasq-5k"
        calibration_path.mkdir(parents=True)
        (calibration_path / "metadata.json").write_text(
            json.dumps({"n_documents": 1}), encoding="utf-8"
        )
        with patch.object(
            bio_io,
            "load_from_disk",
            return_value={"doc_id": ["d1"], "text": ["document"]},
        ):
            calibration = bio_io.load_calibration_set(calibration_path)
        self.assertIsInstance(calibration, CalibrationSet)

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
        self.assertNotIn("show_progress_bar", config_json)

    def test_metric_settings_have_separate_identity(self) -> None:
        pipeline_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "cosine",
            registry_db_path=self.registry_path,
        )
        default_evaluation = evaluation.register_evaluation(
            registry_db_path=self.registry_path,
        )
        changed_evaluation = evaluation.register_evaluation(
            {"ndcg_at_k": (5, 10)},
            registry_db_path=self.registry_path,
        )
        self.assertNotEqual(default_evaluation, changed_evaluation)
        with sqlite3.connect(self.registry_path) as connection:
            pipeline_config = json.loads(
                connection.execute(
                    "SELECT config_json FROM pipelines WHERE pipeline_id = ?",
                    (pipeline_id,),
                ).fetchone()[0]
            )
        self.assertNotIn("metric_config", pipeline_config)

    def test_bm25_and_dense_are_distinct_pipeline_types(self) -> None:
        dense_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "cosine",
            registry_db_path=self.registry_path,
        )
        bm25_id = evaluation.register_pipeline(
            retriever_type=RetrieverType.BM25,
            registry_db_path=self.registry_path,
        )
        self.assertNotEqual(dense_id, bm25_id)
        with registry.open_registry_database(self.registry_path) as connection:
            bm25_definition, _ = registry.get_pipeline(connection, bm25_id)
        self.assertIs(bm25_definition.retriever_type, RetrieverType.BM25)

    def test_compute_embedding_mean_uses_raw_document_embeddings(self) -> None:
        model = _CalibrationModel()
        embedding_mean = evaluation.compute_embedding_mean(
            model,
            ["one", "two"],
            batch_size=2,
            show_progress_bar=False,
        )
        self.assertEqual(embedding_mean, (2.0, 4.0))
        self.assertFalse(model.arguments["normalize_embeddings"])

    def test_evaluate_keys_results_by_pipeline_evaluation_and_dataset(self) -> None:
        dataset_path = self._create_dataset_directory("list-one")
        pipeline_id = evaluation.register_pipeline(
            retriever_type="bm25",
            registry_db_path=self.registry_path,
        )
        evaluation_id = evaluation.register_evaluation(
            registry_db_path=self.registry_path,
        )
        with (
            patch.object(evaluation, "load_bioasq_sample", return_value=_sample_payload()),
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
