"""Append-only SQLite persistence for retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation_models import (
    BioASQSample,
    DatasetRecord,
    DatasetRegistration,
    DatasetValidationError,
    EvaluationRegistryError,
    PipelineDefinition,
    PipelineNotFoundError,
    SimilarityMetric,
)
from src.io import load_bioasq_sample, mount_google_drive

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_ROOT = Path("/content/drive/MyDrive/Retreaval/databases")
DEFAULT_REGISTRY_DB = DEFAULT_DATABASE_ROOT / "datasets.sqlite"
DEFAULT_RESULTS_DB = DEFAULT_DATABASE_ROOT / "results.sqlite"

DEFAULT_METRIC_CONFIG: dict[str, tuple[int, ...]] = {
    "mrr_at_k": (10,),
    "ndcg_at_k": (10,),
    "accuracy_at_k": (1, 3, 5, 10, 100),
    "precision_recall_at_k": (1, 3, 5, 10, 100),
    "map_at_k": (100,),
}

_RESERVED_EVALUATOR_ARGUMENTS = {
    "queries",
    "corpus",
    "relevant_docs",
    "corpus_chunk_size",
    "batch_size",
    "name",
    "show_progress_bar",
    "write_csv",
    "score_functions",
    "main_score_function",
    *DEFAULT_METRIC_CONFIG,
}

_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id TEXT PRIMARY KEY,
    pipeline_hash TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    similarity_metric TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    content_hash TEXT NOT NULL,
    source_path TEXT NOT NULL,
    n_queries INTEGER NOT NULL CHECK (n_queries > 0),
    n_documents INTEGER NOT NULL CHECK (n_documents > 0),
    n_relevance_relations INTEGER NOT NULL CHECK (n_relevance_relations > 0),
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (dataset_name, version),
    UNIQUE (dataset_name, content_hash)
);

CREATE INDEX IF NOT EXISTS datasets_name_version_idx
ON datasets (dataset_name, version DESC);

CREATE TRIGGER IF NOT EXISTS pipelines_prevent_update
BEFORE UPDATE ON pipelines
BEGIN
    SELECT RAISE(ABORT, 'pipelines is append-only');
END;

CREATE TRIGGER IF NOT EXISTS pipelines_prevent_delete
BEFORE DELETE ON pipelines
BEGIN
    SELECT RAISE(ABORT, 'pipelines is append-only');
END;

CREATE TRIGGER IF NOT EXISTS datasets_prevent_update
BEFORE UPDATE ON datasets
BEGIN
    SELECT RAISE(ABORT, 'datasets is append-only');
END;

CREATE TRIGGER IF NOT EXISTS datasets_prevent_delete
BEFORE DELETE ON datasets
BEGIN
    SELECT RAISE(ABORT, 'datasets is append-only');
END;
"""

_RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    result_id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_version INTEGER NOT NULL CHECK (dataset_version > 0),
    dataset_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    similarity_metric TEXT NOT NULL,
    pipeline_config_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    duration_seconds REAL NOT NULL CHECK (duration_seconds >= 0),
    UNIQUE (pipeline_id, dataset_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    result_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (result_id, metric_name),
    FOREIGN KEY (result_id) REFERENCES evaluation_runs (result_id)
);

CREATE INDEX IF NOT EXISTS evaluation_runs_dataset_idx
ON evaluation_runs (dataset_id);

CREATE TRIGGER IF NOT EXISTS evaluation_runs_prevent_update
BEFORE UPDATE ON evaluation_runs
BEGIN
    SELECT RAISE(ABORT, 'evaluation_runs is append-only');
END;

CREATE TRIGGER IF NOT EXISTS evaluation_runs_prevent_delete
BEFORE DELETE ON evaluation_runs
BEGIN
    SELECT RAISE(ABORT, 'evaluation_runs is append-only');
END;

CREATE TRIGGER IF NOT EXISTS metrics_prevent_update
BEFORE UPDATE ON metrics
BEGIN
    SELECT RAISE(ABORT, 'metrics is append-only');
END;

CREATE TRIGGER IF NOT EXISTS metrics_prevent_delete
BEFORE DELETE ON metrics
BEGIN
    SELECT RAISE(ABORT, 'metrics is append-only');
END;
"""


def _utc_now() -> str:
    current_time = datetime.now(timezone.utc)
    return current_time.isoformat()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Configuration must be JSON serializable") from error


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return f"{prefix}_{digest.hexdigest()[:24]}"


@contextmanager
def _open_database(
    database_path: str | Path,
) -> Iterator[sqlite3.Connection]:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA synchronous = FULL")
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@contextmanager
def open_registry_database(
    database_path: str | Path = DEFAULT_REGISTRY_DB,
) -> Iterator[sqlite3.Connection]:
    """Open the registry database and ensure its append-only schema."""
    with _open_database(database_path) as connection:
        connection.executescript(_REGISTRY_SCHEMA)
        yield connection


@contextmanager
def open_results_database(
    database_path: str | Path = DEFAULT_RESULTS_DB,
) -> Iterator[sqlite3.Connection]:
    """Open the results database and ensure its append-only schema."""
    with _open_database(database_path) as connection:
        connection.executescript(_RESULTS_SCHEMA)
        yield connection


def _normalise_metric_config(
    metric_config: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[int, ...]]:
    merged = dict(DEFAULT_METRIC_CONFIG)
    if metric_config is not None:
        unknown = set(metric_config).difference(DEFAULT_METRIC_CONFIG)
        if unknown:
            raise ValueError(
                f"Unsupported metric configuration keys: {sorted(unknown)}"
            )
        merged.update(
            {
                name: tuple(sorted(int(cutoff) for cutoff in cutoffs))
                for name, cutoffs in metric_config.items()
            }
        )

    for name, cutoffs in merged.items():
        if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
            raise ValueError(f"{name} must contain positive integer cutoffs")
        if len(set(cutoffs)) != len(cutoffs):
            raise ValueError(f"{name} contains duplicate cutoffs")
    return merged


def _build_pipeline_definition(
    model_name: str,
    similarity_metric: SimilarityMetric | str,
    batch_size: int,
    corpus_chunk_size: int,
    metric_config: Mapping[str, Sequence[int]] | None,
    model_kwargs: Mapping[str, Any] | None,
    evaluator_kwargs: Mapping[str, Any] | None,
    show_progress_bar: bool,
    embedding_mean: Sequence[float] | None,
) -> PipelineDefinition:
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("model_name must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if corpus_chunk_size <= 0:
        raise ValueError("corpus_chunk_size must be positive")

    resolved_model_kwargs = dict(model_kwargs or {})
    if "device" in resolved_model_kwargs:
        raise ValueError(
            "Pass device to evaluate(); it is runtime metadata, not pipeline identity"
        )

    resolved_evaluator_kwargs = dict(evaluator_kwargs or {})
    reserved = set(resolved_evaluator_kwargs).intersection(
        _RESERVED_EVALUATOR_ARGUMENTS
    )
    if reserved:
        raise ValueError(
            "evaluator_kwargs contains centrally managed arguments: "
            f"{sorted(reserved)}"
        )

    resolved_metric = SimilarityMetric.parse(similarity_metric)
    resolved_embedding_mean: tuple[float, ...] | None = None
    if embedding_mean is not None:
        resolved_embedding_mean = tuple(float(value) for value in embedding_mean)
        if not resolved_embedding_mean:
            raise ValueError("embedding_mean must not be empty")
        if not all(math.isfinite(value) for value in resolved_embedding_mean):
            raise ValueError("embedding_mean must contain only finite values")

    requires_mean = resolved_metric is SimilarityMetric.MEAN_CENTERED_COSINE
    if requires_mean and resolved_embedding_mean is None:
        raise ValueError(
            "mean_centered_cosine requires an embedding_mean"
        )
    if not requires_mean and resolved_embedding_mean is not None:
        raise ValueError(
            "embedding_mean is only valid for mean_centered_cosine"
        )

    definition = PipelineDefinition(
        model_name=model_name,
        similarity_metric=resolved_metric,
        batch_size=batch_size,
        corpus_chunk_size=corpus_chunk_size,
        metric_config=_normalise_metric_config(metric_config),
        model_kwargs=resolved_model_kwargs,
        evaluator_kwargs=resolved_evaluator_kwargs,
        show_progress_bar=show_progress_bar,
        embedding_mean=resolved_embedding_mean,
    )
    _canonical_json(definition.to_dict())
    return definition


def register_pipeline(
    model_name: str,
    similarity_metric: SimilarityMetric | str,
    *,
    registry_db_path: str | Path = DEFAULT_REGISTRY_DB,
    batch_size: int = 64,
    corpus_chunk_size: int = 10_000,
    metric_config: Mapping[str, Sequence[int]] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    evaluator_kwargs: Mapping[str, Any] | None = None,
    show_progress_bar: bool = True,
    embedding_mean: Sequence[float] | None = None,
) -> str:
    """Return the ID of an existing or newly appended pipeline definition."""
    mount_google_drive()
    definition = _build_pipeline_definition(
        model_name=model_name,
        similarity_metric=similarity_metric,
        batch_size=batch_size,
        corpus_chunk_size=corpus_chunk_size,
        metric_config=metric_config,
        model_kwargs=model_kwargs,
        evaluator_kwargs=evaluator_kwargs,
        show_progress_bar=show_progress_bar,
        embedding_mean=embedding_mean,
    )
    config_json = _canonical_json(definition.to_dict())
    pipeline_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    pipeline_id = _stable_id("pipeline", pipeline_hash)

    with open_registry_database(registry_db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT pipeline_id FROM pipelines WHERE pipeline_hash = ?",
            (pipeline_hash,),
        ).fetchone()
        if existing is not None:
            connection.commit()
            return str(existing["pipeline_id"])

        connection.execute(
            """
            INSERT INTO pipelines (
                pipeline_id,
                pipeline_hash,
                model_name,
                similarity_metric,
                config_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                pipeline_id,
                pipeline_hash,
                definition.model_name,
                definition.similarity_metric.value,
                config_json,
                _utc_now(),
            ),
        )
        connection.commit()

    logger.info(
        "Registered pipeline %s for model=%s, similarity=%s",
        pipeline_id,
        definition.model_name,
        definition.similarity_metric.value,
    )
    return pipeline_id


def _update_hash_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big"))
    digest.update(encoded)


def _hash_dataset_content(sample: BioASQSample) -> str:
    digest = hashlib.sha256()

    _update_hash_text(digest, "queries")
    for query_id in sorted(sample.queries):
        _update_hash_text(digest, query_id)
        _update_hash_text(digest, sample.queries[query_id])

    _update_hash_text(digest, "relevant_docs")
    for query_id in sorted(sample.relevant_docs):
        _update_hash_text(digest, query_id)
        for document_id in sorted(sample.relevant_docs[query_id]):
            _update_hash_text(digest, document_id)

    _update_hash_text(digest, "corpus")
    for document_id in sorted(sample.corpus):
        _update_hash_text(digest, document_id)
        _update_hash_text(digest, sample.corpus[document_id])

    return digest.hexdigest()


def _dataset_record_from_row(row: sqlite3.Row) -> DatasetRecord:
    return DatasetRecord(
        dataset_id=str(row["dataset_id"]),
        dataset_name=str(row["dataset_name"]),
        version=int(row["version"]),
        content_hash=str(row["content_hash"]),
        source_path=Path(str(row["source_path"])),
    )


def _select_dataset_record(
    connection: sqlite3.Connection,
    dataset_id: str,
) -> DatasetRecord:
    row = connection.execute(
        """
        SELECT dataset_id, dataset_name, version, content_hash, source_path
        FROM datasets
        WHERE dataset_id = ?
        """,
        (dataset_id,),
    ).fetchone()
    if row is None:
        raise EvaluationRegistryError(f"Unknown dataset_id: {dataset_id}")
    return _dataset_record_from_row(row)


def _select_latest_dataset_record(
    connection: sqlite3.Connection,
    dataset_name: str,
) -> DatasetRecord:
    row = connection.execute(
        """
        SELECT dataset_id, dataset_name, version, content_hash, source_path
        FROM datasets
        WHERE dataset_name = ?
        ORDER BY version DESC
        LIMIT 1
        """,
        (dataset_name,),
    ).fetchone()
    if row is None:
        raise EvaluationRegistryError(f"No registered dataset named {dataset_name!r}")
    return _dataset_record_from_row(row)


def register_loaded_dataset(
    dataset_path: str | Path,
    sample: BioASQSample,
    *,
    registry_db_path: str | Path = DEFAULT_REGISTRY_DB,
) -> DatasetRegistration:
    """Append a loaded dataset version and return current and latest records."""
    resolved_path = Path(dataset_path).resolve()
    if not resolved_path.is_dir():
        raise DatasetValidationError(
            f"Dataset directory does not exist: {resolved_path}"
        )

    dataset_name = resolved_path.name
    content_hash = _hash_dataset_content(sample)
    dataset_id = _stable_id("dataset", dataset_name, content_hash)
    stored_metadata = {
        key: value
        for key, value in sample.metadata.items()
        if key not in {"queries", "relevant_docs"}
    }

    with open_registry_database(registry_db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT dataset_id
            FROM datasets
            WHERE dataset_name = ? AND content_hash = ?
            """,
            (dataset_name, content_hash),
        ).fetchone()
        if existing is None:
            version_row = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next_version
                FROM datasets
                WHERE dataset_name = ?
                """,
                (dataset_name,),
            ).fetchone()
            version = int(version_row["next_version"])
            connection.execute(
                """
                INSERT INTO datasets (
                    dataset_id,
                    dataset_name,
                    version,
                    content_hash,
                    source_path,
                    n_queries,
                    n_documents,
                    n_relevance_relations,
                    metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    dataset_name,
                    version,
                    content_hash,
                    str(resolved_path),
                    len(sample.queries),
                    len(sample.corpus),
                    sum(
                        len(document_ids)
                        for document_ids in sample.relevant_docs.values()
                    ),
                    _canonical_json(stored_metadata),
                    _utc_now(),
                ),
            )
            logger.info(
                "Registered dataset %s as %s version %d",
                dataset_id,
                dataset_name,
                version,
            )
        else:
            dataset_id = str(existing["dataset_id"])

        current = _select_dataset_record(connection, dataset_id)
        latest = _select_latest_dataset_record(connection, dataset_name)
        connection.commit()

    return DatasetRegistration(current=current, latest=latest)


def register_dataset(
    dataset_path: str | Path,
    *,
    registry_db_path: str | Path = DEFAULT_REGISTRY_DB,
) -> str:
    """Validate and append one semantic BioASQ dataset version if necessary."""
    mount_google_drive()
    path = Path(dataset_path)
    try:
        sample = load_bioasq_sample(path)
    except Exception as error:
        raise DatasetValidationError(
            f"Could not load BioASQ dataset at {path}"
        ) from error

    registration = register_loaded_dataset(
        path,
        sample,
        registry_db_path=registry_db_path,
    )
    return registration.current.dataset_id


def get_pipeline(
    connection: sqlite3.Connection,
    pipeline_id: str,
) -> tuple[PipelineDefinition, str]:
    """Load a pipeline definition and its canonical stored JSON."""
    row = connection.execute(
        """
        SELECT config_json
        FROM pipelines
        WHERE pipeline_id = ?
        """,
        (pipeline_id,),
    ).fetchone()
    if row is None:
        raise PipelineNotFoundError(f"Unknown pipeline_id: {pipeline_id}")

    config_json = str(row["config_json"])
    definition = PipelineDefinition.from_dict(json.loads(config_json))
    return definition, config_json


def result_exists(
    connection: sqlite3.Connection,
    pipeline_id: str,
    dataset_id: str,
) -> str | None:
    """Return an existing result ID for one pipeline and dataset."""
    row = connection.execute(
        """
        SELECT result_id
        FROM evaluation_runs
        WHERE pipeline_id = ? AND dataset_id = ?
        """,
        (pipeline_id, dataset_id),
    ).fetchone()
    return None if row is None else str(row["result_id"])


def get_result_metrics(
    connection: sqlite3.Connection,
    result_id: str,
) -> dict[str, float]:
    """Load all stored metrics belonging to one evaluation result."""
    rows = connection.execute(
        """
        SELECT metric_name, value
        FROM metrics
        WHERE result_id = ?
        ORDER BY metric_name
        """,
        (result_id,),
    ).fetchall()
    if not rows:
        raise EvaluationRegistryError(
            f"Result {result_id!r} has no stored metrics"
        )
    return {
        str(row["metric_name"]): float(row["value"])
        for row in rows
    }


def store_evaluation_result(
    connection: sqlite3.Connection,
    pipeline_id: str,
    pipeline_config_json: str,
    definition: PipelineDefinition,
    dataset_record: DatasetRecord,
    started_at: str,
    completed_at: str,
    duration_seconds: float,
    metrics: Mapping[str, float],
) -> str:
    """Atomically append one evaluation run and all of its metrics."""
    result_id = _stable_id(
        "result",
        pipeline_id,
        dataset_record.dataset_id,
    )
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        """
        INSERT INTO evaluation_runs (
            result_id,
            pipeline_id,
            dataset_id,
            dataset_name,
            dataset_version,
            dataset_hash,
            model_name,
            similarity_metric,
            pipeline_config_json,
            started_at,
            completed_at,
            duration_seconds
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result_id,
            pipeline_id,
            dataset_record.dataset_id,
            dataset_record.dataset_name,
            dataset_record.version,
            dataset_record.content_hash,
            definition.model_name,
            definition.similarity_metric.value,
            pipeline_config_json,
            started_at,
            completed_at,
            duration_seconds,
        ),
    )
    connection.executemany(
        """
        INSERT INTO metrics (result_id, metric_name, value)
        VALUES (?, ?, ?)
        """,
        [
            (result_id, metric_name, value)
            for metric_name, value in sorted(metrics.items())
        ],
    )
    connection.commit()
    return result_id
