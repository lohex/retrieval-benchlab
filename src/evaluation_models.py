"""Typed data models for persistent retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class EvaluationRegistryError(RuntimeError):
    """Base exception for persistent retrieval evaluation errors."""


class DatasetValidationError(EvaluationRegistryError):
    """Raised when a directory is not a valid BioASQ retrieval dataset."""


class PipelineNotFoundError(EvaluationRegistryError):
    """Raised when a requested pipeline ID is not registered."""


class SimilarityMetric(str, Enum):
    """Similarity functions supported by Sentence Transformers."""

    COSINE = "cosine"
    MEAN_CENTERED_COSINE = "mean_centered_cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

    @classmethod
    def parse(cls, value: SimilarityMetric | str) -> SimilarityMetric:
        """Normalize a similarity metric and common aliases."""
        if isinstance(value, cls):
            return value

        aliases = {
            "cos": cls.COSINE,
            "cosine": cls.COSINE,
            "centered_cosine": cls.MEAN_CENTERED_COSINE,
            "mean_centered_cosine": cls.MEAN_CENTERED_COSINE,
            "dot": cls.DOT,
            "dot_product": cls.DOT,
            "euclidean": cls.EUCLIDEAN,
            "manhattan": cls.MANHATTAN,
        }
        try:
            return aliases[str(value).strip().lower()]
        except KeyError as error:
            supported = ", ".join(metric.value for metric in cls)
            raise ValueError(
                f"Unsupported similarity metric {value!r}; "
                f"expected one of: {supported}"
            ) from error


class EvaluationStatus(str, Enum):
    """Outcome of evaluating one registered dataset."""

    EVALUATED = "evaluated"
    SKIPPED_EXISTING = "skipped_existing"
    SKIPPED_NOT_LATEST = "skipped_not_latest"


@dataclass(frozen=True)
class BioASQSample:
    """Loaded and validated BioASQ retrieval sample."""

    queries: dict[str, str]
    relevant_docs: dict[str, set[str]]
    corpus: dict[str, str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CalibrationSet:
    """Documents and provenance used for model-specific preprocessing."""

    corpus: dict[str, str]
    metadata: dict[str, Any]


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
    embedding_mean: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        config = {
            "model_name": self.model_name,
            "similarity_metric": self.similarity_metric.value,
            "batch_size": self.batch_size,
            "corpus_chunk_size": self.corpus_chunk_size,
            "metric_config": self.metric_config,
            "model_kwargs": self.model_kwargs,
            "evaluator_kwargs": self.evaluator_kwargs,
            "show_progress_bar": self.show_progress_bar,
        }
        if self.embedding_mean is not None:
            config["embedding_mean"] = self.embedding_mean
        return config

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PipelineDefinition:
        """Construct a pipeline definition from stored JSON."""
        metric_config = {
            key: tuple(int(cutoff) for cutoff in cutoffs)
            for key, cutoffs in value["metric_config"].items()
        }
        return cls(
            model_name=str(value["model_name"]),
            similarity_metric=SimilarityMetric.parse(
                value["similarity_metric"]
            ),
            batch_size=int(value["batch_size"]),
            corpus_chunk_size=int(value["corpus_chunk_size"]),
            metric_config=metric_config,
            model_kwargs=dict(value["model_kwargs"]),
            evaluator_kwargs=dict(value["evaluator_kwargs"]),
            show_progress_bar=bool(value["show_progress_bar"]),
            embedding_mean=(
                tuple(float(item) for item in value["embedding_mean"])
                if value.get("embedding_mean") is not None
                else None
            ),
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
class DatasetRegistration:
    """Current dataset identity and the latest identity with the same name."""

    current: DatasetRecord
    latest: DatasetRecord


@dataclass(frozen=True)
class EvaluationOutcome:
    """Result or skip reason for one dataset considered by ``evaluate``."""

    dataset_id: str
    dataset_name: str
    dataset_version: int
    status: EvaluationStatus
    result_id: str | None = None
    metrics: dict[str, float] | None = None
