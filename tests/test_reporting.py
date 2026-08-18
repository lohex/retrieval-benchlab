"""Unit tests for read-only registry reporting."""

from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.reporting import RegistryReport, load_registry_report


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.registry_path = self.root / "datasets.sqlite"
        self.results_path = self.root / "results.sqlite"
        self._create_registry()
        self._create_results()

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def _create_registry(self) -> None:
        with sqlite3.connect(self.registry_path) as connection:
            connection.executescript(
                """
                CREATE TABLE datasets (
                    dataset_id TEXT,
                    dataset_name TEXT,
                    version INTEGER,
                    content_hash TEXT,
                    source_path TEXT,
                    n_queries INTEGER,
                    n_documents INTEGER,
                    n_relevance_relations INTEGER,
                    metadata_json TEXT,
                    created_at TEXT
                );
                CREATE TABLE pipelines (
                    pipeline_id TEXT,
                    model_name TEXT,
                    similarity_metric TEXT,
                    config_json TEXT,
                    created_at TEXT
                );
                """
            )
            connection.executemany(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        "dataset_old",
                        "list-one",
                        1,
                        "old",
                        "/tmp/list-one",
                        2,
                        10,
                        2,
                        json.dumps({"n_positive_documents": 2}),
                        "2026-01-01",
                    ),
                    (
                        "dataset_new",
                        "list-one",
                        2,
                        "new",
                        "/tmp/list-one",
                        4,
                        20,
                        8,
                        json.dumps({"n_positive_documents": 6}),
                        "2026-01-02",
                    ),
                    (
                        "dataset_factoid",
                        "factoid-multiple",
                        1,
                        "factoid",
                        "/tmp/factoid-multiple",
                        5,
                        30,
                        15,
                        json.dumps({"n_positive_documents": 12}),
                        "2026-01-01",
                    ),
                ],
            )
            connection.executemany(
                "INSERT INTO pipelines VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        "pipeline_aaaaaaaa",
                        "org/model-a",
                        "cosine",
                        "{}",
                        "2026-01-01",
                    ),
                    (
                        "pipeline_bbbbbbbb",
                        "BM25",
                        "bm25",
                        "{}",
                        "2026-01-02",
                    ),
                ],
            )

    def _create_results(self) -> None:
        with sqlite3.connect(self.results_path) as connection:
            connection.executescript(
                """
                CREATE TABLE evaluation_runs (
                    result_id TEXT,
                    pipeline_id TEXT,
                    evaluation_id TEXT,
                    dataset_id TEXT,
                    dataset_name TEXT,
                    dataset_version INTEGER,
                    duration_seconds REAL,
                    completed_at TEXT
                );
                CREATE TABLE metrics (
                    result_id TEXT,
                    metric_name TEXT,
                    value REAL
                );
                """
            )
            connection.executemany(
                "INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        "result_old",
                        "pipeline_aaaaaaaa",
                        "evaluation_default",
                        "dataset_old",
                        "list-one",
                        1,
                        1.0,
                        "2026-01-01",
                    ),
                    (
                        "result_new",
                        "pipeline_aaaaaaaa",
                        "evaluation_default",
                        "dataset_new",
                        "list-one",
                        2,
                        1.0,
                        "2026-01-02",
                    ),
                    (
                        "result_factoid",
                        "pipeline_bbbbbbbb",
                        "evaluation_default",
                        "dataset_factoid",
                        "factoid-multiple",
                        1,
                        1.0,
                        "2026-01-02",
                    ),
                ],
            )
            connection.executemany(
                "INSERT INTO metrics VALUES (?, ?, ?)",
                [
                    ("result_old", "ndcg@10", 0.1),
                    ("result_new", "ndcg@10", 0.8),
                    ("result_factoid", "ndcg@10", 0.5),
                ],
            )

    def test_report_uses_latest_dataset_versions_only(self) -> None:
        report = load_registry_report(
            self.registry_path,
            self.results_path,
        )

        self.assertIsInstance(report, RegistryReport)
        self.assertEqual(len(report.datasets), 2)
        list_row = report.datasets.set_index("dataset").loc["list-one"]
        self.assertEqual(list_row["version"], 2)
        self.assertEqual(list_row["positive_relations_per_query"], 2.0)
        self.assertEqual(list_row["unique_positives_per_query"], 1.5)

        self.assertNotIn("dataset_old", set(report.metrics["dataset_id"]))
        self.assertEqual(set(report.metrics["value"]), {0.8, 0.5})
        self.assertEqual(
            set(report.metrics["evaluation_id"]),
            {"evaluation_default"},
        )
        self.assertEqual(
            set(report.pipelines["evaluated_latest_datasets"]),
            {1},
        )
        self.assertEqual(set(report.pipelines["coverage"]), {0.5})


if __name__ == "__main__":
    unittest.main()
