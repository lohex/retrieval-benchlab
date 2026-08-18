"""Ranking metrics used by all retriever backends."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    retrieved = ranking[:k]
    if not retrieved:
        return 0.0
    hits = sum(document_id in relevant for document_id in retrieved)
    return hits / k


def _recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(document_id in relevant for document_id in ranking[:k])
    return hits / len(relevant)


def _reciprocal_rank(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    for index, document_id in enumerate(ranking[:k], start=1):
        if document_id in relevant:
            return 1.0 / index
    return 0.0


def _average_precision(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, document_id in enumerate(ranking[:k], start=1):
        if document_id in relevant:
            hits += 1
            precision_sum += hits / index
    return precision_sum / min(len(relevant), k)


def _ndcg(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for index, document_id in enumerate(ranking[:k], start=1):
        if document_id in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevant_docs: Mapping[str, set[str]],
    metric_config: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    """Compute macro-averaged IR metrics for ranked document IDs."""
    if set(rankings) != set(relevant_docs):
        raise ValueError("Ranking and relevance query IDs differ")

    results: dict[str, float] = {}
    for metric_name, cutoffs in metric_config.items():
        for cutoff in cutoffs:
            k = int(cutoff)
            values: list[float] = []
            for query_id, ranking in rankings.items():
                relevant = relevant_docs[query_id]
                if metric_name == "mrr_at_k":
                    value = _reciprocal_rank(ranking, relevant, k)
                elif metric_name == "ndcg_at_k":
                    value = _ndcg(ranking, relevant, k)
                elif metric_name == "accuracy_at_k":
                    value = float(any(doc in relevant for doc in ranking[:k]))
                elif metric_name == "precision_recall_at_k":
                    precision = _precision_at_k(ranking, relevant, k)
                    recall = _recall_at_k(ranking, relevant, k)
                    values.append(precision)
                    results.setdefault(f"precision@{k}", 0.0)
                    results.setdefault(f"recall@{k}", 0.0)
                    continue
                elif metric_name == "map_at_k":
                    value = _average_precision(ranking, relevant, k)
                else:
                    raise ValueError(f"Unsupported metric: {metric_name}")
                values.append(value)

            if metric_name == "precision_recall_at_k":
                precisions = [
                    _precision_at_k(rankings[qid], relevant_docs[qid], k)
                    for qid in rankings
                ]
                recalls = [
                    _recall_at_k(rankings[qid], relevant_docs[qid], k)
                    for qid in rankings
                ]
                results[f"precision@{k}"] = _average(precisions)
                results[f"recall@{k}"] = _average(recalls)
            else:
                label = {
                    "mrr_at_k": "mrr",
                    "ndcg_at_k": "ndcg",
                    "accuracy_at_k": "accuracy",
                    "map_at_k": "map",
                }[metric_name]
                results[f"{label}@{k}"] = _average(values)
    return results
