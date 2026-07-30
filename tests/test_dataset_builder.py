"""Unit tests for persistent BioASQ dataset construction."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_dependency_stubs() -> None:
    datasets_module = types.ModuleType("datasets")
    datasets_module.Dataset = object
    datasets_module.load_dataset = lambda *args, **kwargs: None
    datasets_module.load_from_disk = lambda *args, **kwargs: None
    sys.modules.setdefault("datasets", datasets_module)

    tqdm_package = types.ModuleType("tqdm")
    tqdm_package.__path__ = []
    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = lambda iterable, **kwargs: iterable
    sys.modules.setdefault("tqdm", tqdm_package)
    sys.modules.setdefault("tqdm.auto", tqdm_auto)


_install_dependency_stubs()

import src.dataset_builder as builder  # noqa: E402


def _benchmark() -> builder.BioASQBenchmark:
    corpus = {
        f"d{index}": f"document-{index}"
        for index in range(1, 31)
    }
    return (
        {
            "q-list-one": "list one",
            "q-list-multiple": "list multiple",
            "q-yesno": "yes or no",
        },
        {
            "q-list-one": {"d1"},
            "q-list-multiple": {"d2", "d3"},
            "q-yesno": {"d4"},
        },
        {
            "q-list-one": "list",
            "q-list-multiple": "list",
            "q-yesno": "yesno",
        },
        corpus,
        {"corpus_dataset_name": "test/corpus"},
    )


class DatasetBuilderTests(unittest.TestCase):
    def test_create_calibration_set_persists_typed_result(self) -> None:
        with patch.object(builder, "save_calibration_set") as saver:
            created = builder.create_bioasq_calibration_set(
                _benchmark(),
                n_documents=5,
                seed=43,
                output_dir="/tmp/calibration",
                protected_question_types=("list",),
            )

        self.assertIsInstance(created, builder.CreatedCalibrationSet)
        self.assertEqual(created.output_dir, Path("/tmp/calibration"))
        self.assertEqual(len(created.calibration_set.corpus), 5)
        self.assertTrue(
            set(created.calibration_set.corpus).isdisjoint(
                {"d1", "d2", "d3"}
            )
        )
        saver.assert_called_once()

    def test_create_sample_excludes_calibration_and_registers(self) -> None:
        calibration_ids = {"d20", "d21", "d22", "d23", "d24"}
        with (
            patch.object(builder, "save_sample") as saver,
            patch.object(
                builder,
                "register_dataset",
                return_value="dataset_test",
            ) as register,
        ):
            created = builder.create_bioasq_sample(
                _benchmark(),
                question_type="list",
                documents="one",
                subset_name="list-one",
                n_queries=1,
                n_corpus_docs=10,
                seed=42,
                output_root="/tmp/data",
                calibration_document_ids=calibration_ids,
                calibration_set_path="/tmp/calibration",
            )

        self.assertIsInstance(created, builder.CreatedBioASQSample)
        self.assertEqual(created.dataset_id, "dataset_test")
        self.assertEqual(created.output_dir, Path("/tmp/data/list-one"))
        self.assertEqual(len(created.sample.corpus), 10)
        self.assertTrue(
            set(created.sample.corpus).isdisjoint(calibration_ids)
        )
        saver.assert_called_once()
        register.assert_called_once_with(Path("/tmp/data/list-one"))


if __name__ == "__main__":
    unittest.main()
