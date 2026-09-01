"""Embedding-space transformations applied before similarity scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EmbeddingTransformType(str, Enum):
    """Corpus-wide transformations applied to dense embeddings."""

    IDENTITY = "identity"
    MEAN_CENTER = "mean_center"
    VARIANCE_NORMALIZE = "variance_normalize"
    Z_NORMALIZE = "z_normalize"

    @classmethod
    def parse(cls, value: EmbeddingTransformType | str) -> EmbeddingTransformType:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported embedding transform {value!r}; expected one of: {supported}"
            ) from error


@dataclass(frozen=True)
class CalibrationStatistics:
    """Per-dimension statistics estimated from raw calibration embeddings."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    source_id: str

    def __post_init__(self) -> None:
        if not self.mean or len(self.mean) != len(self.std):
            raise ValueError("Calibration mean and std must have the same non-zero length")
        if not self.source_id.strip():
            raise ValueError("Calibration source_id must not be empty")
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Calibration statistics must be finite")
        if np.any(std < 0):
            raise ValueError("Calibration standard deviations must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "std": self.std,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> CalibrationStatistics:
        return cls(
            mean=tuple(float(item) for item in value["mean"]),
            std=tuple(float(item) for item in value["std"]),
            source_id=str(value["source_id"]),
        )


@dataclass(frozen=True)
class EmbeddingTransformConfig:
    """Ranking-relevant configuration for a corpus-wide embedding transform."""

    transform_type: EmbeddingTransformType = EmbeddingTransformType.IDENTITY
    calibration: CalibrationStatistics | None = None
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        needs_calibration = self.transform_type is not EmbeddingTransformType.IDENTITY
        if needs_calibration != (self.calibration is not None):
            raise ValueError(
                "Calibration statistics must be supplied exactly for non-identity transforms"
            )

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"transform_type": self.transform_type.value}
        if self.transform_type is EmbeddingTransformType.IDENTITY:
            return value
        value.update(
            {
                "calibration": self.calibration.to_dict(),
                "epsilon": self.epsilon,
            }
        )
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EmbeddingTransformConfig:
        transform_type = EmbeddingTransformType.parse(value["transform_type"])
        if transform_type is EmbeddingTransformType.IDENTITY:
            return cls(transform_type=transform_type)
        return cls(
            transform_type=transform_type,
            calibration=CalibrationStatistics.from_dict(dict(value["calibration"])),
            epsilon=float(value["epsilon"]),
        )


def _calibration_arrays(
    config: EmbeddingTransformConfig,
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if config.calibration is None:
        raise ValueError("Transformation requires calibration statistics")
    mean = np.asarray(config.calibration.mean, dtype=np.float32)
    std = np.asarray(config.calibration.std, dtype=np.float32)
    if mean.shape != (embedding_dim,) or std.shape != (embedding_dim,):
        raise ValueError("Calibration statistics do not match embedding dimensionality")
    return mean, np.maximum(std, np.float32(config.epsilon))


def transform_embeddings(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    config: EmbeddingTransformConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one corpus-wide transformation to query and document embeddings."""
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("Embedding matrices must be two-dimensional")
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        raise ValueError("Query and document embeddings must have equal dimensions")
    transform_type = config.transform_type
    if transform_type is EmbeddingTransformType.IDENTITY:
        return query_embeddings, document_embeddings

    mean, std = _calibration_arrays(config, query_embeddings.shape[1])
    if transform_type is EmbeddingTransformType.MEAN_CENTER:
        return query_embeddings - mean, document_embeddings - mean
    if transform_type is EmbeddingTransformType.VARIANCE_NORMALIZE:
        return query_embeddings / std, document_embeddings / std
    if transform_type is EmbeddingTransformType.Z_NORMALIZE:
        return (query_embeddings - mean) / std, (document_embeddings - mean) / std
    raise ValueError(f"Unhandled embedding transform: {transform_type.value}")
