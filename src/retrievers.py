"""Retriever implementations shared by benchmark evaluations."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np


class RetrieverType(str, Enum):
    """Retrieval backends supported by the benchmark."""

    DENSE = "dense"
    BM25 = "bm25"

    @classmethod
    def parse(cls, value: RetrieverType | str) -> RetrieverType:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported retriever type {value!r}; expected one of: {supported}"
            ) from error


class Retriever(Protocol):
    """Minimal ranking interface used by the evaluator."""

    def rank(
        self,
        queries: Mapping[str, str],
        corpus: Mapping[str, str],
        *,
        top_k: int,
    ) -> dict[str, list[str]]:
        """Return document IDs ordered from best to worst for every query."""


@dataclass(frozen=True)
class DenseRetrieverConfig:
    """Configuration that changes dense retrieval rankings."""

    model_name: str
    similarity_metric: str = "cosine"
    model_kwargs: dict[str, Any] | None = None
    query_prompt: str | None = None
    embedding_mean: tuple[float, ...] | None = None


@dataclass(frozen=True)
class BM25RetrieverConfig:
    """Configuration that changes BM25 rankings."""

    k1: float = 1.5
    b: float = 0.75


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


class BM25Retriever:
    """Simple lexical baseline backed by ``rank_bm25``."""

    def __init__(self, config: BM25RetrieverConfig) -> None:
        self.config = config

    def rank(
        self,
        queries: Mapping[str, str],
        corpus: Mapping[str, str],
        *,
        top_k: int,
    ) -> dict[str, list[str]]:
        from rank_bm25 import BM25Okapi

        doc_ids = list(corpus)
        tokenized_corpus = [_tokenize(corpus[doc_id]) for doc_id in doc_ids]
        index = BM25Okapi(tokenized_corpus, k1=self.config.k1, b=self.config.b)
        limit = min(top_k, len(doc_ids))
        rankings: dict[str, list[str]] = {}
        for query_id, query in queries.items():
            scores = np.asarray(index.get_scores(_tokenize(query)))
            order = np.argsort(-scores, kind="stable")[:limit]
            rankings[query_id] = [doc_ids[int(index)] for index in order]
        return rankings


class DenseRetriever:
    """Sentence-Transformers dense retriever with optional query instructions."""

    def __init__(
        self,
        config: DenseRetrieverConfig,
        *,
        device: str | None,
        batch_size: int,
        corpus_scan_size: int,
        show_progress_bar: bool,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.batch_size = batch_size
        self.corpus_scan_size = corpus_scan_size
        self.show_progress_bar = show_progress_bar
        self.model = SentenceTransformer(
            config.model_name,
            device=device,
            **(config.model_kwargs or {}),
        )

    def _encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        encode_query = getattr(self.model, "encode_query", self.model.encode)
        prepared = list(texts)
        if self.config.query_prompt:
            prepared = [f"{self.config.query_prompt}{text}" for text in prepared]
        return np.asarray(
            encode_query(
                prepared,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )

    def _encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        encode_document = getattr(self.model, "encode_document", self.model.encode)
        return np.asarray(
            encode_document(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )

    def _prepare_embeddings(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        metric = self.config.similarity_metric
        if metric == "mean_centered_cosine":
            if self.config.embedding_mean is None:
                raise ValueError("mean_centered_cosine requires embedding_mean")
            mean = np.asarray(self.config.embedding_mean, dtype=np.float32)
            query_embeddings = query_embeddings - mean
            corpus_embeddings = corpus_embeddings - mean
        if metric in {"cosine", "mean_centered_cosine"}:
            query_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            corpus_norm = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            query_embeddings = query_embeddings / np.clip(query_norm, 1e-12, None)
            corpus_embeddings = corpus_embeddings / np.clip(corpus_norm, 1e-12, None)
        return query_embeddings, corpus_embeddings

    def _score_block(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        if self.config.similarity_metric == "euclidean":
            return -np.linalg.norm(
                query_embeddings[:, None, :] - corpus_embeddings[None, :, :],
                axis=2,
            )
        if self.config.similarity_metric == "manhattan":
            return -np.abs(
                query_embeddings[:, None, :] - corpus_embeddings[None, :, :]
            ).sum(axis=2)
        return query_embeddings @ corpus_embeddings.T

    def rank(
        self,
        queries: Mapping[str, str],
        corpus: Mapping[str, str],
        *,
        top_k: int,
    ) -> dict[str, list[str]]:
        query_ids = list(queries)
        doc_ids = list(corpus)
        query_embeddings = self._encode_queries([queries[qid] for qid in query_ids])
        corpus_embeddings = self._encode_documents([corpus[doc_id] for doc_id in doc_ids])
        query_embeddings, corpus_embeddings = self._prepare_embeddings(
            query_embeddings,
            corpus_embeddings,
        )
        limit = min(top_k, len(doc_ids))
        best_scores = np.full((len(query_ids), limit), -np.inf, dtype=np.float32)
        best_indices = np.full((len(query_ids), limit), -1, dtype=np.int64)

        for start in range(0, len(doc_ids), self.corpus_scan_size):
            stop = min(start + self.corpus_scan_size, len(doc_ids))
            scores = self._score_block(query_embeddings, corpus_embeddings[start:stop])
            block_indices = np.arange(start, stop, dtype=np.int64)
            candidate_scores = np.concatenate([best_scores, scores], axis=1)
            candidate_indices = np.concatenate(
                [
                    best_indices,
                    np.broadcast_to(block_indices, (len(query_ids), len(block_indices))),
                ],
                axis=1,
            )
            keep = np.argpartition(candidate_scores, -limit, axis=1)[:, -limit:]
            best_scores = np.take_along_axis(candidate_scores, keep, axis=1)
            best_indices = np.take_along_axis(candidate_indices, keep, axis=1)

        rankings: dict[str, list[str]] = {}
        for row_index, query_id in enumerate(query_ids):
            order = np.argsort(-best_scores[row_index], kind="stable")
            rankings[query_id] = [
                doc_ids[int(best_indices[row_index, position])]
                for position in order
            ]
        return rankings
