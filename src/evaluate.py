"""Orchestration of registered document-retrieval evaluations."""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    "evaluate",
    "register_dataset",
    "register_pipeline",
]


def _utc_now() -> str:
    current_time = datetime.now(timezone.utc)
    return current_time.isoformat()


def _build_evaluator(
    definition: PipelineDefinition,
    dataset_record: DatasetRecord,
    sample: BioASQSample,
) -> Any:
    from sentence_transformers import util
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    score_functions = {
        SimilarityMetric.COSINE: util.cos_sim,
        SimilarityMetric.DOT: util.dot_score,
        SimilarityMetric.EUCLIDEAN: util.euclidean_sim,
        SimilarityMetric.MANHATTAN: util.manhattan_sim,
    }
    metric_name = definition.similarity_metric.value
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
        score_functions={
            metric_name: score_functions[definition.similarity_metric]
        },
        main_score_function=metric_name,
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
