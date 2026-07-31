"""Orchestration of registered document-retrieval evaluations."""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation_models import (
    BioASQSample,
    DatasetRecord,
    EvaluationOutcome,
    EvaluationRegistryError,
    EvaluationStatus,
    PipelineDefinition,
    SimilarityMetric,
)
from src.evaluation_registry import (
    DEFAULT_REGISTRY_DB,
    DEFAULT_RESULTS_DB,
    get_pipeline,
    get_result_metrics,
    open_registry_database,
    open_results_database,
    register_dataset,
    register_loaded_dataset,
    register_pipeline,
    result_exists,
    store_evaluation_result,
)
from src.io import (
    discover_dataset_directories,
    load_bioasq_sample,
    mount_google_drive,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BioASQSample",
    "EvaluationOutcome",
    "EvaluationStatus",
    "SimilarityMetric",
    "compute_embedding_mean",
    "evaluate",
    "register_dataset",
    "register_pipeline",
]


def _utc_now() -> str:
    current_time = datetime.now(timezone.utc)
    return current_time.isoformat()


def compute_embedding_mean(
    model: Any,
    documents: Iterable[str],
    *,
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> tuple[float, ...]:
    """Encode calibration documents and return one mean per dimension."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    document_texts = [str(document).strip() for document in documents]
    if not document_texts:
        raise ValueError("documents must not be empty")
    if any(not document for document in document_texts):
        raise ValueError("documents must not contain empty text")

    encode_documents = getattr(model, "encode_document", None)
    if encode_documents is None:
        encode_documents = model.encode
    embeddings = encode_documents(
        document_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embedding_matrix = np.asarray(embeddings)
    has_valid_shape = (
        embedding_matrix.ndim == 2
        and embedding_matrix.shape[0] == len(document_texts)
        and embedding_matrix.shape[1] > 0
    )
    if not has_valid_shape:
        raise EvaluationRegistryError(
            "Model returned an invalid calibration embedding matrix"
        )
    if not np.isfinite(embedding_matrix).all():
        raise EvaluationRegistryError(
            "Calibration embeddings contain non-finite values"
        )

    embedding_mean = embedding_matrix.mean(axis=0, dtype=np.float64)
    logger.info(
        "Computed a %d-dimensional mean from %d calibration documents",
        embedding_mean.shape[0],
        embedding_matrix.shape[0],
    )
    return tuple(float(value) for value in embedding_mean)


def _mean_centered_cosine_score(
    query_embeddings: Any,
    corpus_embeddings: Any,
    *,
    embedding_mean: tuple[float, ...],
    cosine_score: Callable[[Any, Any], Any],
) -> Any:
    query_dimension = int(query_embeddings.shape[-1])
    corpus_dimension = int(corpus_embeddings.shape[-1])
    dimensions_match = (
        query_dimension == corpus_dimension == len(embedding_mean)
    )
    if not dimensions_match:
        raise EvaluationRegistryError(
            "Embedding dimension does not match the stored calibration mean"
        )

    query_mean = query_embeddings.new_tensor(embedding_mean)
    corpus_mean = corpus_embeddings.new_tensor(embedding_mean)
    centered_queries = query_embeddings - query_mean
    centered_corpus = corpus_embeddings - corpus_mean
    return cosine_score(centered_queries, centered_corpus)


def _score_function(
    definition: PipelineDefinition,
    cosine_score: Callable[[Any, Any], Any],
    dot_score: Callable[[Any, Any], Any],
    euclidean_score: Callable[[Any, Any], Any],
    manhattan_score: Callable[[Any, Any], Any],
) -> Callable[[Any, Any], Any]:
    if definition.similarity_metric is SimilarityMetric.MEAN_CENTERED_COSINE:
        if definition.embedding_mean is None:
            raise EvaluationRegistryError(
                "Mean-centered cosine pipeline has no embedding mean"
            )

        def mean_centered_cosine(
            query_embeddings: Any,
            corpus_embeddings: Any,
        ) -> Any:
            return _mean_centered_cosine_score(
                query_embeddings,
                corpus_embeddings,
                embedding_mean=definition.embedding_mean,
                cosine_score=cosine_score,
            )

        return mean_centered_cosine

    score_functions = {
        SimilarityMetric.COSINE: cosine_score,
        SimilarityMetric.DOT: dot_score,
        SimilarityMetric.EUCLIDEAN: euclidean_score,
        SimilarityMetric.MANHATTAN: manhattan_score,
    }
    return score_functions[definition.similarity_metric]


def _build_evaluator(
    definition: PipelineDefinition,
    dataset_record: DatasetRecord,
    sample: BioASQSample,
) -> Any:
    from sentence_transformers import util
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    metric_name = definition.similarity_metric.value
    score_function = _score_function(
        definition,
        cosine_score=util.cos_sim,
        dot_score=util.dot_score,
        euclidean_score=util.euclidean_sim,
        manhattan_score=util.manhattan_sim,
    )
    main_score_function: str | None = metric_name
    if definition.similarity_metric is SimilarityMetric.MEAN_CENTERED_COSINE:
        main_score_function = None
    evaluator_arguments = {
        **definition.metric_config,
        **definition.evaluator_kwargs,
    }
    return InformationRetrievalEvaluator(
        queries=sample.queries,
        corpus=sample.corpus,
        relevant_docs=sample.relevant_docs,
        corpus_chunk_size=definition.corpus_chunk_size,
        batch_size=definition.batch_size,
        name=f"{dataset_record.dataset_name}-v{dataset_record.version}",
        show_progress_bar=definition.show_progress_bar,
        write_csv=False,
        score_functions={metric_name: score_function},
        main_score_function=main_score_function,
        **evaluator_arguments,
    )


def _normalise_metric_name(
    raw_name: str,
    definition: PipelineDefinition,
    dataset_record: DatasetRecord,
) -> str:
    dataset_prefix = (
        f"{dataset_record.dataset_name}-v{dataset_record.version}_"
    )
    metric_name = (
        raw_name.removeprefix(dataset_prefix)
        .removeprefix(f"{definition.similarity_metric.value}_")
    )
    return metric_name


def _normalise_results(
    results: dict[str, Any],
    definition: PipelineDefinition,
    dataset_record: DatasetRecord,
) -> dict[str, float]:
    normalised: dict[str, float] = {}
    for raw_name, value in results.items():
        metric_name = _normalise_metric_name(
            str(raw_name),
            definition,
            dataset_record,
        )
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise EvaluationRegistryError(
                f"Metric {raw_name!r} is not numeric: {value!r}"
            ) from error
        if not math.isfinite(numeric_value):
            raise EvaluationRegistryError(
                f"Metric {raw_name!r} is not finite: {numeric_value}"
            )
        if metric_name in normalised:
            raise EvaluationRegistryError(
                f"Metric-name normalization produced duplicate {metric_name!r}"
            )
        normalised[metric_name] = numeric_value

    if not normalised:
        raise EvaluationRegistryError("Evaluator returned no metrics")
    return normalised


def _existing_outcome(
    connection: sqlite3.Connection,
    pipeline_id: str,
    dataset_record: DatasetRecord,
) -> EvaluationOutcome | None:
    result_id = result_exists(
        connection,
        pipeline_id,
        dataset_record.dataset_id,
    )
    if result_id is None:
        return None
    metrics = get_result_metrics(connection, result_id)
    return EvaluationOutcome(
        dataset_id=dataset_record.dataset_id,
        dataset_name=dataset_record.dataset_name,
        dataset_version=dataset_record.version,
        status=EvaluationStatus.SKIPPED_EXISTING,
        result_id=result_id,
        metrics=metrics,
    )


def _evaluate_sample(
    connection: sqlite3.Connection,
    pipeline_id: str,
    pipeline_config_json: str,
    definition: PipelineDefinition,
    dataset_record: DatasetRecord,
    sample: BioASQSample,
    model: Any,
) -> EvaluationOutcome:
    evaluator = _build_evaluator(
        definition,
        dataset_record,
        sample,
    )
    started_at = _utc_now()
    start_time = time.perf_counter()
    raw_results = evaluator(model)
    duration_seconds = time.perf_counter() - start_time
    completed_at = _utc_now()
    metrics = _normalise_results(
        raw_results,
        definition,
        dataset_record,
    )

    try:
        result_id = store_evaluation_result(
            connection=connection,
            pipeline_id=pipeline_id,
            pipeline_config_json=pipeline_config_json,
            definition=definition,
            dataset_record=dataset_record,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            metrics=metrics,
        )
    except sqlite3.IntegrityError:
        connection.rollback()
        existing = _existing_outcome(
            connection,
            pipeline_id,
            dataset_record,
        )
        if existing is None:
            raise
        return existing

    logger.info(
        "Stored %d metrics for %s version %d",
        len(metrics),
        dataset_record.dataset_name,
        dataset_record.version,
    )
    return EvaluationOutcome(
        dataset_id=dataset_record.dataset_id,
        dataset_name=dataset_record.dataset_name,
        dataset_version=dataset_record.version,
        status=EvaluationStatus.EVALUATED,
        result_id=result_id,
        metrics=metrics,
    )


def _not_latest_outcome(
    current: DatasetRecord,
    latest: DatasetRecord,
) -> EvaluationOutcome:
    logger.warning(
        "Skipping %s because its current content is not registry version %d",
        current.dataset_name,
        latest.version,
    )
    return EvaluationOutcome(
        dataset_id=current.dataset_id,
        dataset_name=current.dataset_name,
        dataset_version=current.version,
        status=EvaluationStatus.SKIPPED_NOT_LATEST,
    )


def evaluate(
    pipeline_id: str,
    datasets_root: str | Path,
    *,
    registry_db_path: str | Path = DEFAULT_REGISTRY_DB,
    results_db_path: str | Path = DEFAULT_RESULTS_DB,
    device: str | None = None,
) -> list[EvaluationOutcome]:
    """Evaluate a registered pipeline on each latest dataset in a folder.

    Every sample is loaded exactly once during this call, registered by its
    semantic content hash, checked against the latest version with the same
    folder name, and evaluated only when no result exists for the same
    ``pipeline_id`` and ``dataset_id``.
    """
    mount_google_drive()
    registry_path = Path(registry_db_path).resolve()
    results_path = Path(results_db_path).resolve()
    if registry_path == results_path:
        raise ValueError("registry_db_path and results_db_path must differ")

    with open_registry_database(registry_path) as connection:
        definition, pipeline_config_json = get_pipeline(
            connection,
            pipeline_id,
        )

    dataset_paths = discover_dataset_directories(datasets_root)
    outcomes: list[EvaluationOutcome] = []
    model: Any | None = None

    with open_results_database(results_path) as results_connection:
        for dataset_path in dataset_paths:
            sample = load_bioasq_sample(dataset_path)
            registration = register_loaded_dataset(
                dataset_path,
                sample,
                registry_db_path=registry_path,
            )
            if registration.current.dataset_id != registration.latest.dataset_id:
                outcomes.append(
                    _not_latest_outcome(
                        registration.current,
                        registration.latest,
                    )
                )
                continue

            existing = _existing_outcome(
                results_connection,
                pipeline_id,
                registration.current,
            )
            if existing is not None:
                outcomes.append(existing)
                continue

            if model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    definition.model_name,
                    device=device,
                    **definition.model_kwargs,
                )
                logger.info(
                    "Loaded model %s for pipeline %s",
                    definition.model_name,
                    pipeline_id,
                )

            outcomes.append(
                _evaluate_sample(
                    connection=results_connection,
                    pipeline_id=pipeline_id,
                    pipeline_config_json=pipeline_config_json,
                    definition=definition,
                    dataset_record=registration.current,
                    sample=sample,
                    model=model,
                )
            )

    return outcomes
