"""Unit tests for BioASQ sampling helpers."""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np


def _install_tqdm_stub() -> None:
    tqdm_package = types.ModuleType("tqdm")
    tqdm_package.__path__ = []
    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = lambda iterable, **kwargs: iterable
    sys.modules.setdefault("tqdm", tqdm_package)
    sys.modules.setdefault("tqdm.auto", tqdm_auto)


_install_tqdm_stub()

from src.sampling import (  # noqa: E402
    add_random_negative_documents,
    sample_calibration_documents,
)


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_corpus = {
            f"d{index}": f"document-{index}"
            for index in range(1, 13)
        }

    def test_calibration_excludes_relevant_documents_of_protected_types(
        self,
    ) -> None:
        relevant_docs = {
            "q-list": {"d1"},
            "q-factoid": {"d2", "d3"},
            "q-summary": {"d4"},
            "q-yesno": {"d5"},
        }
        query_types = {
            "q-list": "list",
            "q-factoid": "factoid",
            "q-summary": "summary",
            "q-yesno": "yesno",
        }

        calibration = sample_calibration_documents(
            source_corpus=self.source_corpus,
            relevant_docs=relevant_docs,
            query_types=query_types,
            n_documents=5,
            rng=np.random.default_rng(43),
        )

        self.assertEqual(len(calibration), 5)
        self.assertTrue(set(calibration).isdisjoint({"d1", "d2", "d3", "d4"}))

    def test_negative_sampling_excludes_calibration_documents(self) -> None:
        calibration_ids = {"d8", "d9", "d10"}
        corpus = add_random_negative_documents(
            source_corpus=self.source_corpus,
            relevant_docs={"q1": {"d1"}},
            target_size=6,
            rng=np.random.default_rng(42),
            excluded_document_ids=calibration_ids,
        )

        self.assertEqual(len(corpus), 6)
        self.assertIn("d1", corpus)
        self.assertTrue(set(corpus).isdisjoint(calibration_ids))

    def test_negative_sampling_rejects_excluded_positive(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "overlap with relevant documents",
        ):
            add_random_negative_documents(
                source_corpus=self.source_corpus,
                relevant_docs={"q1": {"d1"}},
                target_size=6,
                rng=np.random.default_rng(42),
                excluded_document_ids={"d1"},
            )


if __name__ == "__main__":
    unittest.main()
