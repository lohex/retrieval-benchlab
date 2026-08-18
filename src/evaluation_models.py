"""Typed data models for persistent retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.retrievers import RetrieverType


class EvaluationRegistryError(RuntimeError):
    """Base exception for persistent retrieval evaluation errors."""


class DatasetValidationError(EvaluationRegistryError):
    """Raised when a directory is not a valid BioASQ retrieval dataset."""


class PipelineNotFoundError(EvaluationRegistryError):
    """Raised when a requested pipeline ID is not registered."""


class EvaluationNotFoundError(EvaluationRegistryError):
    """Raised when a requested evaluation definition is not registered."""


class SimilarityMetric(str, Enum):
    """Similarity functions supported by dense retrieval pipelines."""

    COSINE = "cosine"
    MEAN_CENTERED_COSINE = "mean_centered_cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

    @classmethod
    def parse(cls, value: SimilarityMetric | str) -> SimilarityMetric:
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
                f"Unsupported similarity metric {value!r}; expected one of: {supported}"
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
    """Immutable configuration that can change retrieval rankings."""

    retriever_type: RetrieverType
    model_name: str | None = None
    similarity_metric: SimilarityMetric | None = None
    model_kwargs: dict[str, Any] | None = None
    query_prompt: str | None = None
    embedding_mean: tuple[float, ...] | None = None
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "retriever_type": self.retriever_type.value,
        }
        if self.retriever_type is RetrieverType.DENSE:
            config.update(
                {
                    "model_name": self.model_name,
                    "similarity_metric": (
                        self.similarity_metric.value
                        if self.similarity_metric is not None
                        else None
                    ),
                    "model_kwargs": self.model_kwargs or {},
                    "query_prompt": self.query_prompt,
                }
            )
            if self.embedding_mean is not None:
                config["embedding_mean"] = self.embedding_mean
        else:
            config.update({"bm25_k1": self.bm25_k1, "bm25_b": self.bm25_b})
        return config

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PipelineDefinition:
        retriever_type = RetrieverType.parse(value["retriever_type"])
        if retriever_type is RetrieverType.BM25:
            return cls(
                retriever_type=retriever_type,
                bm25_k1=float(value["bm25_k1"]),
                bm25_b=float(value["bm25_b"]),
            )
        return cls(
            retriever_type=retriever_type,
            model_name=str(value["model_name"]),
            similarity_metric=SimilarityMetric.parse(value["similarity_metric"]),
            model_kwargs=dict(value["model_kwargs"]),
            query_prompt=(
                str(value["query_prompt"])
                if value["query_prompt"] is not None
                else None
            ),
            embedding_mean=(
                tuple(float(item) for item in value["embedding_mean"])
                if "embedding_mean" in value
                else None
            ),
        )


@dataclass(frozen=True)
class EvaluationDefinition:
    """Immutable metric configuration independent of the retriever."""

    metric_config: dict[str, tuple[int, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {"metric_config": self.metric_config}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationDefinition:
        return cls(
            metric_config={
                key: tuple(int(cutoff) for cutoff in cutoffs)
                for key, cutoffs in value["metric_config"].items()
            }
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution-only settings that never affect benchmark identities."""

    batch_size: int = 64
    corpus_scan_size: int = 10_000
    show_progress_bar: bool = True
    device: str | None = None


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
    """Result or skip reason for one registered dataset."""

    dataset_id: str
    dataset_name: str
    dataset_version: int
    status: EvaluationStatus
    result_id: str | None = None
    metrics: dict[str, float] | None = None
