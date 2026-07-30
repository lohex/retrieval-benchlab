"""Orchestration for creating persistent BioASQ benchmark datasets."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Collection, Set
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation_models import BioASQSample, CalibrationSet
from src.evaluation_registry import register_dataset
from src.io import (
    BioASQBenchmark,
    sample_directory,
    save_calibration_set,
    save_sample,
)
from src.sampling import (
    RETRIEVAL_QUESTION_TYPES,
    add_random_negative_documents,
    sample_calibration_documents,
    sample_queries_and_positives,
    validate_sample,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CreatedBioASQSample:
    """A newly persisted and registered BioASQ retrieval sample."""

    sample: BioASQSample
    output_dir: Path
    dataset_id: str


@dataclass(frozen=True)
class CreatedCalibrationSet:
    """A newly persisted document calibration set."""

    calibration_set: CalibrationSet
    output_dir: Path


def create_bioasq_calibration_set(
    benchmark: BioASQBenchmark,
    *,
    n_documents: int,
    seed: int,
    output_dir: str | Path,
    protected_question_types: Collection[str] = RETRIEVAL_QUESTION_TYPES,
) -> CreatedCalibrationSet:
    """Create and persist a shared BioASQ document calibration set."""
    (
        _,
        source_relevant_docs,
        source_query_types,
        source_corpus,
        source_metadata,
    ) = benchmark
    normalized_types = tuple(
        sorted(question_type.lower() for question_type in protected_question_types)
    )
    calibration_corpus = sample_calibration_documents(
        source_corpus=source_corpus,
        relevant_docs=source_relevant_docs,
        query_types=source_query_types,
        n_documents=n_documents,
        rng=np.random.default_rng(seed),
        protected_question_types=normalized_types,
    )
    protected_relevant_ids = set().union(
        *(
            source_relevant_docs[query_id]
            for query_id, question_type in source_query_types.items()
            if question_type in normalized_types
        )
    )
    metadata = {
        **source_metadata,
        "set_name": Path(output_dir).name,
        "purpose": "embedding calibration",
        "n_documents": len(calibration_corpus),
        "seed": seed,
        "protected_question_types": list(normalized_types),
        "n_protected_relevant_documents": len(protected_relevant_ids),
        "sampling": (
            "uniform sample excluding protected relevant documents"
        ),
    }
    resolved_output_dir = Path(output_dir)
    save_calibration_set(
        resolved_output_dir,
        calibration_corpus,
        metadata,
    )
    calibration_set = CalibrationSet(
        corpus=calibration_corpus,
        metadata=metadata,
    )
    return CreatedCalibrationSet(
        calibration_set=calibration_set,
        output_dir=resolved_output_dir,
    )


def create_bioasq_sample(
    benchmark: BioASQBenchmark,
    *,
    question_type: str | None,
    documents: str,
    subset_name: str | None,
    n_queries: int | None,
    n_corpus_docs: int,
    seed: int,
    output_root: str | Path,
    calibration_document_ids: Set[str] | None = None,
    calibration_set_path: str | Path | None = None,
) -> CreatedBioASQSample:
    """Create, persist, and register one filtered BioASQ retrieval sample."""
    if n_corpus_docs <= 0:
        raise ValueError("n_corpus_docs must be positive")
    normalized_question_type = (
        question_type.lower() if question_type else None
    )
    normalized_documents = documents.lower()
    (
        source_queries,
        source_relevant_docs,
        source_query_types,
        source_corpus,
        source_metadata,
    ) = benchmark
    rng = np.random.default_rng(seed)

    logger.info(
        "Selecting questions for type=%s and documents=%s",
        normalized_question_type or "all",
        normalized_documents,
    )
    queries, relevant_docs, query_types, n_eligible = (
        sample_queries_and_positives(
            queries=source_queries,
            relevant_docs=source_relevant_docs,
            query_types=source_query_types,
            n_queries=n_queries,
            rng=rng,
            question_type=normalized_question_type,
            documents=normalized_documents,
        )
    )

    logger.info("Sampling random negative documents")
    corpus = add_random_negative_documents(
        source_corpus=source_corpus,
        relevant_docs=relevant_docs,
        target_size=n_corpus_docs,
        rng=rng,
        excluded_document_ids=calibration_document_ids,
    )
    validate_sample(
        queries,
        relevant_docs,
        corpus,
        expected_queries=len(queries),
        expected_corpus_docs=n_corpus_docs,
    )

    type_label = normalized_question_type or "all"
    resolved_subset_name = subset_name
    if resolved_subset_name is None:
        if type_label == normalized_documents == "all":
            resolved_subset_name = "current"
        else:
            resolved_subset_name = (
                f"{type_label}-{normalized_documents}"
            )
    output_dir = sample_directory(output_root, resolved_subset_name)
    positive_ids = set().union(*relevant_docs.values())
    metadata = {
        **source_metadata,
        "subset_name": resolved_subset_name,
        "question_type_filter": type_label,
        "documents_filter": normalized_documents,
        "n_eligible_queries": n_eligible,
        "n_queries": len(queries),
        "n_corpus_docs": len(corpus),
        "n_positive_documents": len(positive_ids),
        "n_positive_relations": sum(
            len(document_ids)
            for document_ids in relevant_docs.values()
        ),
        "seed": seed,
        "query_type_counts": dict(
            sorted(Counter(query_types.values()).items())
        ),
        "query_types": query_types,
        "calibration_set_path": (
            str(calibration_set_path)
            if calibration_set_path is not None
            else None
        ),
        "n_calibration_documents": len(
            calibration_document_ids or ()
        ),
        "sampling": (
            "filtered expert questions plus all gold documents and "
            "uniformly sampled corpus negatives"
        ),
    }
    save_sample(
        output_dir,
        queries,
        relevant_docs,
        corpus,
        metadata,
    )
    dataset_id = register_dataset(output_dir)
    logger.info("Dataset registered as %s", dataset_id)
    sample = BioASQSample(
        queries=queries,
        relevant_docs=relevant_docs,
        corpus=corpus,
        metadata=metadata,
    )
    return CreatedBioASQSample(
        sample=sample,
        output_dir=output_dir,
        dataset_id=dataset_id,
    )
