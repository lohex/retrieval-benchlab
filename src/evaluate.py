"""Orchestration of registered document-retrieval evaluations."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation_models import (
    BioASQSample,
    DatasetRecord,
    EvaluationDefinition,
    EvaluationOutcome,
    EvaluationStatus,
    PipelineDefinition,
    RuntimeConfig,
    SimilarityMetric,
)
from src.evaluation_registry import (
    DEFAULT_REGISTRY_DB,
    DEFAULT_RESULTS_DB,
    get_evaluation,
    get_pipeline,
    get_result_metrics,
    open_registry_database,
    open_results_database,
    register_dataset,
    register_evaluation,
    register_loaded_dataset,
    register_pipeline,
    result_exists,
    store_evaluation_result,
)
from src.io import discover_dataset_directories, load_bioasq_sample, mount_google_drive
from src.metrics import evaluate_rankings
from src.retrievers import (
    BM25Retriever,
    BM25RetrieverConfig,
    DenseRetriever,
    DenseRetrieverConfig,
    Retriever,
    RetrieverType,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BioASQSample",
    "EvaluationOutcome",
    "EvaluationStatus",
    "RuntimeConfig",
    "SimilarityMetric",
    "compute_embedding_mean",
    "evaluate",
    "register_dataset",
    "register_evaluation",
    "register_pipeline",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    valid_shape = (
        embedding_matrix.ndim == 2
        and embedding_matrix.shape[0] == len(document_texts)
        and embedding_matrix.shape[1] > 0
    )
    if not valid_shape:
        raise ValueError("Model returned an invalid calibration embedding matrix")
    if not np.isfinite(embedding_matrix).all():
        raise ValueError("Calibration embeddings contain non-finite values")
    embedding_mean = embedding_matrix.mean(axis=0, dtype=np.float64)
    return tuple(float(value) for value in embedding_mean)


def _max_metric_cutoff(definition: EvaluationDefinition) -> int:
    return max(
        cutoff
        for cutoffs in definition.metric_config.values()
        for cutoff in cutoffs
    )


def _build_retriever(
    definition: PipelineDefinition,
    runtime: RuntimeConfig,
) -> Retriever:
    if definition.retriever_type is RetrieverType.BM25:
        return BM25Retriever(
            BM25RetrieverConfig(
                k1=definition.bm25_k1,
                b=definition.bm25_b,
            )
        )
    if definition.model_name is None or definition.similarity_metric is None:
        raise ValueError("Dense pipeline is missing model configuration")
    return DenseRetriever(
        DenseRetrieverConfig(
            model_name=definition.model_name,
            similarity_metric=definition.similarity_metric.value,
            model_kwargs=definition.model_kwargs,
            query_prompt=definition.query_prompt,
            embedding_mean=definition.embedding_mean,
        ),
        device=runtime.device,
        batch_size=runtime.batch_size,
        corpus_scan_size=runtime.corpus_scan_size,
        show_progress_bar=runtime.show_progress_bar,
    )


def _existing_outcome(
    connection: sqlite3.Connection,
    pipeline_id: str,
    evaluation_id: str,
    dataset_record: DatasetRecord,
) -> EvaluationOutcome | None:
    result_id = result_exists(
        connection,
        pipeline_id,
        evaluation_id,
        dataset_record.dataset_id,
    )
    if result_id is None:
        return None
    return EvaluationOutcome(
        dataset_id=dataset_record.dataset_id,
        dataset_name=dataset_record.dataset_name,
        dataset_version=dataset_record.version,
        status=EvaluationStatus.SKIPPED_EXISTING,
        result_id=result_id,
        metrics=get_result_metrics(connection, result_id),
    )


def _evaluate_sample(
    connection: sqlite3.Connection,
    *,
    pipeline_id: str,
    evaluation_id: str,
    pipeline_config_json: str,
    evaluation_config_json: str,
    evaluation_definition: EvaluationDefinition,
    dataset_record: DatasetRecord,
    sample: BioASQSample,
    retriever: Retriever,
) -> EvaluationOutcome:
    started_at = _utc_now()
    start_time = time.perf_counter()
    rankings = retriever.rank(
        sample.queries,
        sample.corpus,
        top_k=_max_metric_cutoff(evaluation_definition),
    )
    metrics = evaluate_rankings(
        rankings,
        sample.relevant_docs,
        evaluation_definition.metric_config,
    )
    duration_seconds = time.perf_counter() - start_time
    completed_at = _utc_now()
    try:
        result_id = store_evaluation_result(
            connection,
            pipeline_id=pipeline_id,
            evaluation_id=evaluation_id,
            pipeline_config_json=pipeline_config_json,
            evaluation_config_json=evaluation_config_json,
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
            evaluation_id,
            dataset_record,
        )
        if existing is None:
            raise
        return existing
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
    evaluation_id: str | None = None,
    runtime: RuntimeConfig | None = None,
    registry_db_path: str | Path = DEFAULT_REGISTRY_DB,
    results_db_path: str | Path = DEFAULT_RESULTS_DB,
) -> list[EvaluationOutcome]:
    """Evaluate one retrieval pipeline on each latest dataset in a folder."""
    mount_google_drive()
    runtime = runtime or RuntimeConfig()
    if runtime.batch_size <= 0:
        raise ValueError("runtime.batch_size must be positive")
    if runtime.corpus_scan_size <= 0:
        raise ValueError("runtime.corpus_scan_size must be positive")

    registry_path = Path(registry_db_path).resolve()
    results_path = Path(results_db_path).resolve()
    if registry_path == results_path:
        raise ValueError("registry_db_path and results_db_path must differ")
    if evaluation_id is None:
        evaluation_id = register_evaluation(registry_db_path=registry_path)

    with open_registry_database(registry_path) as connection:
        pipeline_definition, pipeline_config_json = get_pipeline(
            connection,
            pipeline_id,
        )
        evaluation_definition, evaluation_config_json = get_evaluation(
            connection,
            evaluation_id,
        )

    dataset_paths = discover_dataset_directories(datasets_root)
    outcomes: list[EvaluationOutcome] = []
    retriever: Retriever | None = None
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
                    _not_latest_outcome(registration.current, registration.latest)
                )
                continue
            existing = _existing_outcome(
                results_connection,
                pipeline_id,
                evaluation_id,
                registration.current,
            )
            if existing is not None:
                outcomes.append(existing)
                continue
            if retriever is None:
                retriever = _build_retriever(pipeline_definition, runtime)
            outcomes.append(
                _evaluate_sample(
                    results_connection,
                    pipeline_id=pipeline_id,
                    evaluation_id=evaluation_id,
                    pipeline_config_json=pipeline_config_json,
                    evaluation_config_json=evaluation_config_json,
                    evaluation_definition=evaluation_definition,
                    dataset_record=registration.current,
                    sample=sample,
                    retriever=retriever,
                )
            )
    return outcomes
