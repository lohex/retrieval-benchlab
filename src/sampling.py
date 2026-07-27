"""Sampling helpers for BioASQ retrieval experiments."""

from __future__ import annotations

import logging

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def combine_title_text(title: str | None, text: str | None) -> str:
    """Combine a document title and body into the retrievable text."""
    title = (title or "").strip()
    text = (text or "").strip()
    return f"{title}\n{text}".strip() if title else text


def sample_queries_and_positives(
    dataset,
    n_queries: int,
    rng,
    max_rounds: int = 20,
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, str]]:
    """Sample non-empty queries with unique positive document IDs."""
    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    positive_corpus: dict[str, str] = {}
    used_source_indices: set[int] = set()
    used_positive_ids: set[str] = set()

    progress = tqdm(total=n_queries, desc="Sampling queries", unit="query")
    for round_number in range(1, max_rounds + 1):
        if len(queries) >= n_queries:
            break

        remaining = n_queries - len(queries)
        candidate_count = min(len(dataset), max(remaining * 3, 200))
        candidate_indices = rng.choice(
            len(dataset), size=candidate_count, replace=False
        )
        rows = dataset.select(candidate_indices.tolist())

        accepted = 0
        for source_index, row in zip(candidate_indices, rows):
            source_index = int(source_index)
            doc_id = str(row["_id"])
            query = row["query"]
            document = combine_title_text(row["title"], row["text"])
            if (
                source_index in used_source_indices
                or doc_id in used_positive_ids
                or not isinstance(query, str)
                or not query.strip()
                or not document
            ):
                continue

            qid = f"query-{source_index}"
            used_source_indices.add(source_index)
            used_positive_ids.add(doc_id)
            queries[qid] = query.strip()
            relevant_docs[qid] = {doc_id}
            positive_corpus[doc_id] = document
            accepted += 1
            progress.update(1)
            if len(queries) >= n_queries:
                break

        logger.info(
            "Query round %d: accepted %d, total %d/%d",
            round_number,
            accepted,
            len(queries),
            n_queries,
        )

    progress.close()
    if len(queries) < n_queries:
        raise RuntimeError(
            f"Only {len(queries)} of {n_queries} queries could be sampled"
        )
    return queries, relevant_docs, positive_corpus


def add_random_negative_documents(
    dataset,
    corpus: dict[str, str],
    target_size: int,
    rng,
    max_rounds: int = 20,
) -> dict[str, str]:
    """Add random unique documents while preserving every positive document."""
    corpus = dict(corpus)
    target_size = max(target_size, len(corpus))
    progress = tqdm(
        total=target_size,
        initial=len(corpus),
        desc="Sampling corpus",
        unit="doc",
    )

    for round_number in range(1, max_rounds + 1):
        if len(corpus) >= target_size:
            break

        remaining = target_size - len(corpus)
        candidate_count = min(len(dataset), max(remaining * 2, 1_000))
        candidate_indices = rng.choice(
            len(dataset), size=candidate_count, replace=False
        )
        rows = dataset.select(candidate_indices.tolist())

        accepted = 0
        for row in rows:
            doc_id = str(row["_id"])
            if doc_id in corpus:
                continue
            document = combine_title_text(row["title"], row["text"])
            if document:
                corpus[doc_id] = document
                accepted += 1
                progress.update(1)
            if len(corpus) >= target_size:
                break

        logger.info(
            "Corpus round %d: accepted %d, total %d/%d",
            round_number,
            accepted,
            len(corpus),
            target_size,
        )

    progress.close()
    if len(corpus) < target_size:
        raise RuntimeError(
            f"Only {len(corpus)} of {target_size} documents could be sampled"
        )
    return corpus


def validate_sample(
    queries: dict[str, str],
    relevant_docs: dict[str, set[str]],
    corpus: dict[str, str],
    expected_queries: int,
    expected_corpus_docs: int,
) -> None:
    """Validate the structural invariants of a generated retrieval sample."""
    if len(queries) != expected_queries:
        raise ValueError(f"Expected {expected_queries} queries, found {len(queries)}")
    if len(corpus) != max(expected_corpus_docs, expected_queries):
        raise ValueError(f"Unexpected corpus size: {len(corpus)}")
    if set(queries) != set(relevant_docs):
        raise ValueError("Query IDs and relevance IDs differ")

    positive_ids = set().union(*relevant_docs.values())
    if not positive_ids.issubset(corpus):
        raise ValueError("At least one positive document is missing from the corpus")
    if any(not text.strip() for text in queries.values()):
        raise ValueError("At least one query is empty")
    if any(not text.strip() for text in corpus.values()):
        raise ValueError("At least one corpus document is empty")

    logger.info(
        "Validated sample: %d queries, %d positives, %d corpus documents",
        len(queries),
        len(positive_ids),
        len(corpus),
    )
