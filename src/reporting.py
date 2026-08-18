"""Read-only reporting helpers for retrieval registries."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class ReportingError(RuntimeError):
    """Raised when registry data cannot produce a consistent report."""


@dataclass(frozen=True)
class RegistryReport:
    """Tables used by the registry visualization notebook."""

    datasets: pd.DataFrame
    pipelines: pd.DataFrame
    metrics: pd.DataFrame


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _read_query(database_path: Path, query: str) -> pd.DataFrame:
    with _connect_read_only(database_path) as connection:
        return pd.read_sql_query(query, connection)


def _table_exists(database_path: Path, table_name: str) -> bool:
    with _connect_read_only(database_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
    return row is not None


def _latest_datasets(registry_path: Path) -> pd.DataFrame:
    datasets = _read_query(
        registry_path,
        """
        WITH latest_versions AS (
            SELECT dataset_name, MAX(version) AS version
            FROM datasets GROUP BY dataset_name
        )
        SELECT datasets.*
        FROM datasets
        INNER JOIN latest_versions
            ON datasets.dataset_name = latest_versions.dataset_name
            AND datasets.version = latest_versions.version
        ORDER BY datasets.dataset_name
        """,
    )
    if datasets.empty:
        raise ReportingError("The dataset registry contains no datasets")
    return datasets


def _metadata_value(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    return None if value is None else int(value)


def _enrich_dataset_statistics(datasets: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in datasets.to_dict(orient="records"):
        try:
            metadata = json.loads(str(row["metadata_json"]))
        except (TypeError, ValueError) as error:
            raise ReportingError(
                f"Invalid metadata JSON for dataset {row['dataset_id']}"
            ) from error
        n_queries = int(row["n_queries"])
        n_documents = int(row["n_documents"])
        n_relations = int(row["n_relevance_relations"])
        n_positive_documents = _metadata_value(metadata, "n_positive_documents")
        unique_positives_per_query = None
        n_negative_documents = None
        if n_positive_documents is not None:
            unique_positives_per_query = n_positive_documents / n_queries
            n_negative_documents = n_documents - n_positive_documents
        records.append(
            {
                "dataset_id": str(row["dataset_id"]),
                "dataset": str(row["dataset_name"]),
                "version": int(row["version"]),
                "documents": n_documents,
                "queries": n_queries,
                "unique_positive_documents": n_positive_documents,
                "negative_documents": n_negative_documents,
                "positive_relations": n_relations,
                "positive_relations_per_query": n_relations / n_queries,
                "unique_positives_per_query": unique_positives_per_query,
                "created_at": str(row["created_at"]),
            }
        )
    return pd.DataFrame.from_records(records)


def _registered_pipelines(registry_path: Path) -> pd.DataFrame:
    pipelines = _read_query(
        registry_path,
        """
        SELECT pipeline_id, model_name, similarity_metric, config_json, created_at
        FROM pipelines ORDER BY created_at, pipeline_id
        """,
    )
    if pipelines.empty:
        raise ReportingError("The dataset registry contains no pipelines")
    model_short_names = pipelines["model_name"].str.rsplit("/", n=1).str[-1]
    pipeline_suffixes = pipelines["pipeline_id"].str[-6:]
    pipelines["pipeline_label"] = (
        model_short_names
        + " | "
        + pipelines["similarity_metric"]
        + " | "
        + pipeline_suffixes
    )
    return pipelines


def _result_tables(results_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if _table_exists(results_path, "evaluation_runs_v2"):
        runs = _read_query(
            results_path,
            """
            SELECT result_id, pipeline_id, evaluation_id, dataset_id,
                   dataset_name, dataset_version, duration_seconds, completed_at
            FROM evaluation_runs_v2
            """,
        )
        metrics = _read_query(
            results_path,
            "SELECT result_id, metric_name, value FROM metrics_v2",
        )
        if not runs.empty:
            return runs, metrics
    if not _table_exists(results_path, "evaluation_runs"):
        return pd.DataFrame(), pd.DataFrame()
    runs = _read_query(
        results_path,
        """
        SELECT result_id, pipeline_id, 'legacy' AS evaluation_id, dataset_id,
               dataset_name, dataset_version, duration_seconds, completed_at
        FROM evaluation_runs
        """,
    )
    metrics = _read_query(
        results_path,
        "SELECT result_id, metric_name, value FROM metrics",
    )
    return runs, metrics


def _pipeline_coverage(
    pipelines: pd.DataFrame,
    runs: pd.DataFrame,
    latest_dataset_ids: set[str],
) -> pd.DataFrame:
    if runs.empty:
        coverage = pd.Series(dtype=int, name="evaluated_latest_datasets")
    else:
        latest_runs = runs[runs["dataset_id"].isin(latest_dataset_ids)]
        coverage = (
            latest_runs.groupby("pipeline_id")["dataset_id"]
            .nunique()
            .rename("evaluated_latest_datasets")
        )
    result = pipelines.merge(
        coverage,
        how="left",
        left_on="pipeline_id",
        right_index=True,
        validate="one_to_one",
    )
    result["evaluated_latest_datasets"] = (
        result["evaluated_latest_datasets"].fillna(0).astype(int)
    )
    result["available_latest_datasets"] = len(latest_dataset_ids)
    result["coverage"] = (
        result["evaluated_latest_datasets"] / result["available_latest_datasets"]
    )
    return result


def _latest_metric_values(
    datasets: pd.DataFrame,
    pipelines: pd.DataFrame,
    runs: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "pipeline_id",
        "evaluation_id",
        "pipeline_label",
        "model_name",
        "similarity_metric",
        "dataset_id",
        "dataset",
        "version",
        "metric",
        "value",
    ]
    if runs.empty or metrics.empty:
        return pd.DataFrame(columns=columns)
    dataset_keys = datasets[["dataset_id", "dataset_name", "version"]]
    latest_runs = runs.merge(
        dataset_keys,
        how="inner",
        on="dataset_id",
        validate="many_to_one",
    )
    metric_values = latest_runs.merge(
        metrics,
        how="inner",
        on="result_id",
        validate="one_to_many",
    )
    pipeline_keys = pipelines[
        ["pipeline_id", "pipeline_label", "model_name", "similarity_metric"]
    ]
    metric_values = metric_values.merge(
        pipeline_keys,
        how="inner",
        on="pipeline_id",
        validate="many_to_one",
    )
    metric_values = metric_values.rename(
        columns={"dataset_name_y": "dataset", "metric_name": "metric"}
    )
    return metric_values[columns].sort_values(
        ["metric", "pipeline_label", "evaluation_id", "dataset"]
    )


def load_registry_report(
    registry_db_path: str | Path,
    results_db_path: str | Path,
) -> RegistryReport:
    """Load latest dataset statistics and stored pipeline metrics."""
    registry_path = Path(registry_db_path).resolve()
    results_path = Path(results_db_path).resolve()
    if not registry_path.is_file():
        raise ReportingError(f"Registry database does not exist: {registry_path}")
    if not results_path.is_file():
        raise ReportingError(f"Results database does not exist: {results_path}")
    if registry_path == results_path:
        raise ReportingError("registry_db_path and results_db_path must differ")
    latest_datasets = _latest_datasets(registry_path)
    dataset_statistics = _enrich_dataset_statistics(latest_datasets)
    registered_pipelines = _registered_pipelines(registry_path)
    runs, metrics = _result_tables(results_path)
    latest_dataset_ids = set(latest_datasets["dataset_id"].astype(str))
    return RegistryReport(
        datasets=dataset_statistics,
        pipelines=_pipeline_coverage(
            registered_pipelines,
            runs,
            latest_dataset_ids,
        ),
        metrics=_latest_metric_values(
            latest_datasets,
            registered_pipelines,
            runs,
            metrics,
        ),
    )
