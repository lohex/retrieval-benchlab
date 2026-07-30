"""Persistent registration and evaluation of document-retrieval pipelines."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

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


class EvaluationRegistryError(RuntimeError):
    """Base exception for persistent retrieval evaluation errors."""


class DatasetValidationError(EvaluationRegistryError):
    """Raised when a directory is not a valid BioASQ retrieval dataset."""


class PipelineNotFoundError(EvaluationRegistryError):
    """Raised when a requested pipeline ID is not registered."""


class SimilarityMetric(str, Enum):
    """Similarity functions supported by Sentence Transformers."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class EvaluationStatus(str, Enum):
    """Outcome of evaluating one registered dataset."""

    EVALUATED = "evaluated"
    SKIPPED_EXISTING = "skipped_existing"
    SKIPPED_NOT_LATEST = "skipped_not_latest"


@dataclass(frozen=True)
class PipelineDefinition:
    """Immutable configuration used to construct one retrieval pipeline."""

    model_name: str
    similarity_metric: SimilarityMetric
    batch_size: int
    corpus_chunk_size: int
    metric_config: dict[str, tuple[int, ...]]
    model_kwargs: dict[str, Any]
    evaluator_kwargs: dict[str, Any]
    show_progress_bar: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "model_name": self.model_name,
            "similarity_metric": self.similarity_metric.value,
            "batch_size": self.batch_size,
            "corpus_chunk_size": self.corpus_chunk_size,
            "metric_config": self.metric_config,
            "model_kwargs": self.model_kwargs,
            "evaluator_kwargs": self.evaluator_kwargs,
            "show_progress_bar": self.show_progress_bar,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PipelineDefinition:
        """Construct a pipeline definition from stored JSON."""
        metric_config = {
            key: tuple(int(cutoff) for cutoff in cutoffs)
            for key, cutoffs in value["metric_config"].items()
        }
        return cls(
            model_name=str(value["model_name"]),
            similarity_metric=_normalise_similarity_metric(
                value["similarity_metric"]
            ),
            batch_size=int(value["batch_size"]),
            corpus_chunk_size=int(value["corpus_chunk_size"]),
            metric_config=metric_config,
            model_kwargs=dict(value["model_kwargs"]),
            evaluator_kwargs=dict(value["evaluator_kwargs"]),
            show_progress_bar=bool(value["show_progress_bar"]),
        )


@dataclass(frozen=True)
class DatasetRecord:
    """Identity and provenance of one registered dataset version."""

    dataset_id: str
    dataset_name: str
    version: int
    content_hash: str
    source_path: Path


@dataclass(frozen=True)
class EvaluationOutcome:
    """Result or skip reason for one dataset considered by ``evaluate``."""

    dataset_id: str
    dataset_name: str
    dataset_version: int
    status: EvaluationStatus
    result_id: str | None = None
    metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class _CurrentDataset:
    path: Path
    registered: DatasetRecord
    latest: DatasetRecord


@dataclass(frozen=True)
class _RegistryState:
    definition: PipelineDefinition
    pipeline_config_json: str
    datasets: dict[str, _CurrentDataset]


@dataclass(frozen=True)
class _EvaluationPlan:
    outcomes: list[EvaluationOutcome]
    pending: list[_CurrentDataset]


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


def _initialise_registry(connection: sqlite3.Connection) -> None:
    schema = _REGISTRY_SCHEMA
    connection.executescript(schema)


def _initialise_results(connection: sqlite3.Connection) -> None:
    schema = _RESULTS_SCHEMA
    connection.executescript(schema)


def _normalise_similarity_metric(
    similarity_metric: SimilarityMetric | str,
) -> SimilarityMetric:
    if isinstance(similarity_metric, SimilarityMetric):
        return similarity_metric

    aliases = {
        "cos": SimilarityMetric.COSINE,
        "cosine": SimilarityMetric.COSINE,
        "dot": SimilarityMetric.DOT,
        "dot_product": SimilarityMetric.DOT,
        "euclidean": SimilarityMetric.EUCLIDEAN,
        "manhattan": SimilarityMetric.MANHATTAN,
    }
    try:
        return aliases[str(similarity_metric).strip().lower()]
    except KeyError as error:
        supported = ", ".join(metric.value for metric in SimilarityMetric)
        raise ValueError(
            f"Unsupported similarity metric {similarity_metric!r}; "
            f"expected one of: {supported}"
        ) from error


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

    definition = PipelineDefinition(
        model_name=model_name,
        similarity_metric=_normalise_similarity_metric(similarity_metric),
        batch_size=batch_size,
        corpus_chunk_size=corpus_chunk_size,
        metric_config=_normalise_metric_config(metric_config),
        model_kwargs=resolved_model_kwargs,
        evaluator_kwargs=resolved_evaluator_kwargs,
        show_progress_bar=show_progress_bar,
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
    )
    config_json = _canonical_json(definition.to_dict())
    pipeline_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    pipeline_id = _stable_id("pipeline", pipeline_hash)

    with _open_database(registry_db_path) as connection:
        _initialise_registry(connection)
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT pipeline_id FROM pipelines WHERE pipeline_hash = ?",
            (pipeline_hash,),
        ).fetchone()
        if row is not None:
            connection.commit()
            return str(row["pipeline_id"])

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


def _hash_dataset_content(
    queries: Mapping[str, str],
    relevant_docs: Mapping[str, set[str]],
    corpus: Mapping[str, str],
) -> str:
    digest = hashlib.sha256()

    _update_hash_text(digest, "queries")
    for query_id in sorted(queries):
        _update_hash_text(digest, query_id)
        _update_hash_text(digest, queries[query_id])

    _update_hash_text(digest, "relevant_docs")
    for query_id in sorted(relevant_docs):
        _update_hash_text(digest, query_id)
        for document_id in sorted(relevant_docs[query_id]):
            _update_hash_text(digest, document_id)

    _update_hash_text(digest, "corpus")
    for document_id in sorted(corpus):
        _update_hash_text(digest, document_id)
        _update_hash_text(digest, corpus[document_id])

    return digest.hexdigest()


def _load_and_hash_dataset(
    dataset_path: str | Path,
) -> tuple[
    Path,
    str,
    dict[str, str],
    dict[str, set[str]],
    dict[str, str],
    dict[str, Any],
]:
    path = Path(dataset_path)
    if not path.is_dir():
        raise DatasetValidationError(f"Dataset directory does not exist: {path}")
    if not (path / "metadata.json").is_file() or not (path / "corpus").is_dir():
        raise DatasetValidationError(
            f"{path} is not a saved BioASQ sample: "
            "expected metadata.json and corpus/"
        )

    try:
        queries, relevant_docs, corpus, metadata = load_bioasq_sample(path)
    except Exception as error:
        raise DatasetValidationError(
            f"Could not load BioASQ dataset at {path}"
        ) from error

    content_hash = _hash_dataset_content(queries, relevant_docs, corpus)
    return (
        path.resolve(),
        content_hash,
        queries,
        relevant_docs,
        corpus,
        metadata,
    )


def _dataset_record_from_row(row: sqlite3.Row) -> DatasetRecord:
    return DatasetRecord(
        dataset_id=str(row["dataset_id"]),
        dataset_name=str(row["dataset_name"]),
        version=int(row["version"]),
        content_hash=str(row["content_hash"]),
        source_path=Path(str(row["source_path"])),
    )


def register_dataset(
    dataset_path: str | Path,
    *,
    registry_db_path: str | Path = DEFAULT_REGISTRY_DB,
) -> str:
    """Validate and append one semantic BioASQ dataset version if necessary."""
    mount_google_drive()
    (
        resolved_path,
        content_hash,
        queries,
        relevant_docs,
        corpus,
        metadata,
    ) = _load_and_hash_dataset(dataset_path)
    dataset_name = resolved_path.name
    dataset_id = _stable_id("dataset", dataset_name, content_hash)
    stored_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"queries", "relevant_docs"}
    }

    with _open_database(registry_db_path) as connection:
        _initialise_registry(connection)
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT dataset_id
            FROM datasets
            WHERE dataset_name = ? AND content_hash = ?
            """,
            (dataset_name, content_hash),
        ).fetchone()
        if existing is not None:
            connection.commit()
            return str(existing["dataset_id"])

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
                len(queries),
                len(corpus),
                sum(len(ids) for ids in relevant_docs.values()),
                _canonical_json(stored_metadata),
                _utc_now(),
            ),
        )
        connection.commit()

    logger.info(
        "Registered dataset %s as %s version %d",
        dataset_id,
        dataset_name,
        version,
    )
    return dataset_id


def _get_pipeline(
    connection: sqlite3.Connection,
    pipeline_id: str,
) -> tuple[PipelineDefinition, str]:
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


def _get_dataset_record(
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


def _get_latest_dataset_record(
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


def _discover_dataset_directories(datasets_root: str | Path) -> list[Path]:
    root = Path(datasets_root)
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset root does not exist: {root}")

    dataset_paths = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "metadata.json").is_file()
        and (path / "corpus").is_dir()
    )
    if not dataset_paths:
        raise DatasetValidationError(
            f"No saved BioASQ datasets found directly below {root}"
        )
    return dataset_paths


def _result_exists(
    connection: sqlite3.Connection,
    pipeline_id: str,
    dataset_id: str,
) -> str | None:
    row = connection.execute(
        """
        SELECT result_id
        FROM evaluation_runs
        WHERE pipeline_id = ? AND dataset_id = ?
        """,
        (pipeline_id, dataset_id),
    ).fetchone()
    return None if row is None else str(row["result_id"])


def _build_evaluator(
    definition: PipelineDefinition,
    dataset_record: DatasetRecord,
    queries: dict[str, str],
    relevant_docs: dict[str, set[str]],
    corpus: dict[str, str],
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
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
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
    results: Mapping[str, Any],
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


def _store_evaluation_result(
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


def _register_current_datasets(
    datasets_root: str | Path,
    registry_path: Path,
) -> dict[str, tuple[Path, str]]:
    current_datasets: dict[str, tuple[Path, str]] = {}
    for dataset_path in _discover_dataset_directories(datasets_root):
        dataset_id = register_dataset(
            dataset_path,
            registry_db_path=registry_path,
        )
        current_datasets[dataset_path.name] = (dataset_path, dataset_id)
    return current_datasets


def _load_registry_state(
    pipeline_id: str,
    current_datasets: Mapping[str, tuple[Path, str]],
    registry_path: Path,
) -> _RegistryState:
    with _open_database(registry_path) as connection:
        _initialise_registry(connection)
        definition, pipeline_config_json = _get_pipeline(
            connection,
            pipeline_id,
        )
        datasets = {
            dataset_name: _CurrentDataset(
                path=dataset_path,
                registered=_get_dataset_record(connection, dataset_id),
                latest=_get_latest_dataset_record(connection, dataset_name),
            )
            for dataset_name, (dataset_path, dataset_id)
            in current_datasets.items()
        }
    return _RegistryState(
        definition=definition,
        pipeline_config_json=pipeline_config_json,
        datasets=datasets,
    )


def _plan_evaluations(
    connection: sqlite3.Connection,
    pipeline_id: str,
    datasets: Mapping[str, _CurrentDataset],
) -> _EvaluationPlan:
    outcomes: list[EvaluationOutcome] = []
    pending: list[_CurrentDataset] = []

    for dataset_name, current in datasets.items():
        if current.registered.dataset_id != current.latest.dataset_id:
            logger.warning(
                "Skipping %s because its current content is not registry "
                "version %d",
                dataset_name,
                current.latest.version,
            )
            outcomes.append(
                EvaluationOutcome(
                    dataset_id=current.registered.dataset_id,
                    dataset_name=current.registered.dataset_name,
                    dataset_version=current.registered.version,
                    status=EvaluationStatus.SKIPPED_NOT_LATEST,
                )
            )
            continue

        result_id = _result_exists(
            connection,
            pipeline_id,
            current.latest.dataset_id,
        )
        if result_id is not None:
            outcomes.append(
                EvaluationOutcome(
                    dataset_id=current.latest.dataset_id,
                    dataset_name=current.latest.dataset_name,
                    dataset_version=current.latest.version,
                    status=EvaluationStatus.SKIPPED_EXISTING,
                    result_id=result_id,
                )
            )
            continue
        pending.append(current)

    return _EvaluationPlan(outcomes=outcomes, pending=pending)


def _evaluate_dataset(
    connection: sqlite3.Connection,
    pipeline_id: str,
    state: _RegistryState,
    current: _CurrentDataset,
    model: Any,
) -> EvaluationOutcome:
    queries, relevant_docs, corpus, _ = load_bioasq_sample(current.path)
    evaluator = _build_evaluator(
        state.definition,
        current.latest,
        queries,
        relevant_docs,
        corpus,
    )
    started_at = _utc_now()
    start_time = time.perf_counter()
    raw_results = evaluator(model)
    duration_seconds = time.perf_counter() - start_time
    completed_at = _utc_now()
    metrics = _normalise_results(
        raw_results,
        state.definition,
        current.latest,
    )

    try:
        result_id = _store_evaluation_result(
            connection=connection,
            pipeline_id=pipeline_id,
            pipeline_config_json=state.pipeline_config_json,
            definition=state.definition,
            dataset_record=current.latest,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            metrics=metrics,
        )
    except sqlite3.IntegrityError:
        connection.rollback()
        result_id = _result_exists(
            connection,
            pipeline_id,
            current.latest.dataset_id,
        )
        if result_id is None:
            raise
        return EvaluationOutcome(
            dataset_id=current.latest.dataset_id,
            dataset_name=current.latest.dataset_name,
            dataset_version=current.latest.version,
            status=EvaluationStatus.SKIPPED_EXISTING,
            result_id=result_id,
        )

    logger.info(
        "Stored %d metrics for %s version %d",
        len(metrics),
        current.latest.dataset_name,
        current.latest.version,
    )
    return EvaluationOutcome(
        dataset_id=current.latest.dataset_id,
        dataset_name=current.latest.dataset_name,
        dataset_version=current.latest.version,
        status=EvaluationStatus.EVALUATED,
        result_id=result_id,
        metrics=metrics,
    )


def evaluate(
    pipeline_id: str,
    datasets_root: str | Path,
    *,
    registry_db_path: str | Path = DEFAULT_REGISTRY_DB,
    results_db_path: str | Path = DEFAULT_RESULTS_DB,
    device: str | None = None,
) -> list[EvaluationOutcome]:
    """Evaluate a registered pipeline on every latest dataset below a folder.

    Each immediate child with ``metadata.json`` and ``corpus/`` is registered
    first. Evaluation is performed only when that physical dataset is also the
    latest registered version of its folder name and no result exists for the
    same ``pipeline_id`` and ``dataset_id``.
    """
    mount_google_drive()
    registry_path = Path(registry_db_path).resolve()
    results_path = Path(results_db_path).resolve()
    if registry_path == results_path:
        raise ValueError("registry_db_path and results_db_path must differ")

    current_datasets = _register_current_datasets(
        datasets_root,
        registry_path,
    )
    state = _load_registry_state(
        pipeline_id,
        current_datasets,
        registry_path,
    )

    with _open_database(results_path) as results_connection:
        _initialise_results(results_connection)
        plan = _plan_evaluations(
            results_connection,
            pipeline_id,
            state.datasets,
        )
        if not plan.pending:
            return plan.outcomes

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            state.definition.model_name,
            device=device,
            **state.definition.model_kwargs,
        )
        logger.info(
            "Loaded model %s for pipeline %s",
            state.definition.model_name,
            pipeline_id,
        )
        evaluated = [
            _evaluate_dataset(
                connection=results_connection,
                pipeline_id=pipeline_id,
                state=state,
                current=current,
                model=model,
            )
            for current in plan.pending
        ]

    return [*plan.outcomes, *evaluated]
