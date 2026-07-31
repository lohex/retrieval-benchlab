"""Unit tests for the persistent retrieval evaluation registry."""

from __future__ import annotations

import json
import sqlite3
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


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
from src.evaluation_models import BioASQSample, CalibrationSet


class _FakeEvaluator:
    def __init__(self, *, name: str, main_score_function: str, **kwargs) -> None:
        self.metric_prefix = f"{name}_{main_score_function}_"

    def __call__(self, model) -> dict[str, float]:
        return {
            f"{self.metric_prefix}ndcg@10": 0.75,
            f"{self.metric_prefix}map@100": 0.50,
        }


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name


def _sentence_transformers_stubs() -> dict[str, types.ModuleType]:
    package = types.ModuleType("sentence_transformers")
    package.__path__ = []
    package.SentenceTransformer = _FakeSentenceTransformer
    package.util = types.SimpleNamespace(
        cos_sim=object(),
        dot_score=object(),
        euclidean_sim=object(),
        manhattan_sim=object(),
    )

    nested_package = types.ModuleType("sentence_transformers.sentence_transformer")
    nested_package.__path__ = []
    evaluator_module = types.ModuleType(
        "sentence_transformers.sentence_transformer.evaluation"
    )
    evaluator_module.InformationRetrievalEvaluator = _FakeEvaluator
    return {
        "sentence_transformers": package,
        "sentence_transformers.sentence_transformer": nested_package,
        "sentence_transformers.sentence_transformer.evaluation": evaluator_module,
    }


def _sample_payload(content_marker: str = "v1") -> BioASQSample:
    return BioASQSample(
        queries={"q1": "query"},
        relevant_docs={"q1": {"d1"}},
        corpus={
            "d1": f"relevant-{content_marker}",
            "d2": "background",
        },
        metadata={
            "subset_name": "sample",
            "n_queries": 1,
            "n_corpus_docs": 2,
        },
    )


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

    def test_public_api_is_reexported_from_evaluate(self) -> None:
        self.assertIs(evaluation.register_pipeline, registry.register_pipeline)
        self.assertIs(evaluation.register_dataset, registry.register_dataset)

    def test_io_loader_returns_a_typed_sample(self) -> None:
        dataset_path = self._create_dataset_directory("list-one")
        metadata = {
            "queries": {"q1": "query"},
            "relevant_docs": {"q1": ["d1"]},
            "n_queries": 1,
            "n_corpus_docs": 1,
        }
        (dataset_path / "metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        corpus_dataset = {
            "doc_id": ["d1"],
            "text": ["document"],
        }

        with patch.object(
            bio_io,
            "load_from_disk",
            return_value=corpus_dataset,
        ):
            sample = bio_io.load_bioasq_sample(dataset_path)

        self.assertIsInstance(sample, BioASQSample)
        self.assertEqual(sample.queries, {"q1": "query"})
        self.assertEqual(sample.relevant_docs, {"q1": {"d1"}})

    def test_io_loader_returns_a_typed_calibration_set(self) -> None:
        calibration_path = self.root / "calibration" / "bioasq-5k"
        calibration_path.mkdir(parents=True)
        metadata = {
            "set_name": "bioasq-5k",
            "n_documents": 2,
        }
        (calibration_path / "metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        corpus_dataset = {
            "doc_id": ["d1", "d2"],
            "text": ["document one", "document two"],
        }

        with patch.object(
            bio_io,
            "load_from_disk",
            return_value=corpus_dataset,
        ):
            calibration = bio_io.load_calibration_set(calibration_path)

        self.assertIsInstance(calibration, CalibrationSet)
        self.assertEqual(set(calibration.corpus), {"d1", "d2"})
        self.assertEqual(calibration.metadata["n_documents"], 2)

    def test_register_pipeline_is_idempotent_and_append_only(self) -> None:
        first_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "cosine",
            registry_db_path=self.registry_path,
        )
        second_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            evaluation.SimilarityMetric.COSINE,
            registry_db_path=self.registry_path,
        )
        dot_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "dot",
            registry_db_path=self.registry_path,
        )

        self.assertEqual(first_id, second_id)
        self.assertNotEqual(first_id, dot_id)
        with sqlite3.connect(self.registry_path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM pipelines"
            ).fetchone()[0]
            self.assertEqual(count, 2)
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE pipelines SET model_name = 'changed'"
                )

    def test_register_dataset_increments_versions_for_changed_content(self) -> None:
        dataset_path = self._create_dataset_directory("list-multiple")
        samples = iter(
            (
                _sample_payload("v1"),
                _sample_payload("v1"),
                _sample_payload("v2"),
            )
        )

        with patch.object(
            registry,
            "load_bioasq_sample",
            side_effect=lambda path: next(samples),
        ):
            first_id = evaluation.register_dataset(
                dataset_path,
                registry_db_path=self.registry_path,
            )
            repeated_id = evaluation.register_dataset(
                dataset_path,
                registry_db_path=self.registry_path,
            )
            second_id = evaluation.register_dataset(
                dataset_path,
                registry_db_path=self.registry_path,
            )

        self.assertEqual(first_id, repeated_id)
        self.assertNotEqual(first_id, second_id)
        with sqlite3.connect(self.registry_path) as connection:
            versions = connection.execute(
                """
                SELECT version
                FROM datasets
                WHERE dataset_name = 'list-multiple'
                ORDER BY version
                """
            ).fetchall()
        self.assertEqual(versions, [(1,), (2,)])

    def test_evaluate_loads_once_stores_once_and_skips_existing(self) -> None:
        dataset_path = self._create_dataset_directory("list-multiple")
        pipeline_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "dot",
            registry_db_path=self.registry_path,
        )

        module_stubs = _sentence_transformers_stubs()
        with (
            patch.dict(sys.modules, module_stubs),
            patch.object(
                evaluation,
                "load_bioasq_sample",
                return_value=_sample_payload(),
            ) as loader,
        ):
            first = evaluation.evaluate(
                pipeline_id,
                dataset_path.parent,
                registry_db_path=self.registry_path,
                results_db_path=self.results_path,
            )
            self.assertEqual(loader.call_count, 1)

            second = evaluation.evaluate(
                pipeline_id,
                dataset_path.parent,
                registry_db_path=self.registry_path,
                results_db_path=self.results_path,
            )
            self.assertEqual(loader.call_count, 2)

        self.assertEqual(first[0].status, evaluation.EvaluationStatus.EVALUATED)
        self.assertEqual(
            second[0].status,
            evaluation.EvaluationStatus.SKIPPED_EXISTING,
        )
        self.assertEqual(first[0].metrics, {"map@100": 0.5, "ndcg@10": 0.75})
        self.assertEqual(second[0].metrics, first[0].metrics)

        with sqlite3.connect(self.results_path) as connection:
            run_count = connection.execute(
                "SELECT COUNT(*) FROM evaluation_runs"
            ).fetchone()[0]
            metrics = connection.execute(
                "SELECT metric_name, value FROM metrics ORDER BY metric_name"
            ).fetchall()
        self.assertEqual(run_count, 1)
        self.assertEqual(metrics, [("map@100", 0.5), ("ndcg@10", 0.75)])

    def test_evaluate_skips_a_reverted_non_latest_dataset(self) -> None:
        dataset_path = self._create_dataset_directory("list-multiple")
        pipeline_id = evaluation.register_pipeline(
            "sentence-transformers/test-model",
            "cosine",
            registry_db_path=self.registry_path,
        )
        registry.register_loaded_dataset(
            dataset_path,
            _sample_payload("v1"),
            registry_db_path=self.registry_path,
        )
        registry.register_loaded_dataset(
            dataset_path,
            _sample_payload("v2"),
            registry_db_path=self.registry_path,
        )

        with patch.object(
            evaluation,
            "load_bioasq_sample",
            return_value=_sample_payload("v1"),
        ):
            outcomes = evaluation.evaluate(
                pipeline_id,
                dataset_path.parent,
                registry_db_path=self.registry_path,
                results_db_path=self.results_path,
            )

        self.assertEqual(
            outcomes[0].status,
            evaluation.EvaluationStatus.SKIPPED_NOT_LATEST,
        )
        self.assertEqual(outcomes[0].dataset_version, 1)


if __name__ == "__main__":
    unittest.main()
