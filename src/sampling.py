"""Sampling helpers for BioASQ retrieval experiments."""

from __future__ import annotations

import logging

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

VALID_QUESTION_TYPES = {"yesno", "factoid", "list", "summary"}
VALID_DOCUMENT_FILTERS = {"all", "one", "multiple"}


def _normalise_question_type(question_type: str | None) -> str | None:
    if question_type is None:
        return None
    question_type = question_type.lower()
    if question_type == "all":
        return None
    if question_type not in VALID_QUESTION_TYPES:
        raise ValueError(
            f"question_type must be one of {sorted(VALID_QUESTION_TYPES)} or None"
        )
    return question_type


def _normalise_document_filter(documents: str) -> str:
    documents = documents.lower()
    if documents not in VALID_DOCUMENT_FILTERS:
        raise ValueError(
            f"documents must be one of {sorted(VALID_DOCUMENT_FILTERS)}"
        )
    return documents


def filter_query_ids(
    relevant_docs: dict[str, set[str]],
    query_types: dict[str, str],
    question_type: str | None = None,
    documents: str = "all",
) -> list[str]:
    """Filter query IDs by BioASQ question type and gold-document count."""
    question_type = _normalise_question_type(question_type)
    documents = _normalise_document_filter(documents)
    if set(relevant_docs) != set(query_types):
        raise ValueError("Relevance and question-type identifiers differ")

    selected_ids = []
    for qid in sorted(relevant_docs):
        document_count = len(relevant_docs[qid])
        type_matches = question_type is None or query_types[qid] == question_type
        document_matches = (
            documents == "all"
            or (documents == "one" and document_count == 1)
            or (documents == "multiple" and document_count > 1)
        )
        if type_matches and document_matches:
            selected_ids.append(qid)
    return selected_ids


def sample_queries_and_positives(
    queries: dict[str, str],
    relevant_docs: dict[str, set[str]],
    query_types: dict[str, str],
    n_queries: int | None,
    rng,
    question_type: str | None = None,
    documents: str = "all",
) -> tuple[
    dict[str, str],
    dict[str, set[str]],
    dict[str, str],
    int,
]:
    """Filter and optionally sample expert questions and their gold documents."""
    if not (set(queries) == set(relevant_docs) == set(query_types)):
        raise ValueError("Question, relevance, and type identifiers differ")

    eligible_ids = filter_query_ids(
        relevant_docs,
        query_types,
        question_type=question_type,
        documents=documents,
    )
    n_eligible = len(eligible_ids)
    if n_eligible == 0:
        raise ValueError(
            f"No queries match question_type={question_type!r}, "
            f"documents={documents!r}"
        )
    if n_queries is None:
        selected_ids = eligible_ids
    else:
        if n_queries <= 0:
            raise ValueError("n_queries must be positive or None")
        if n_queries > n_eligible:
            raise ValueError(
                f"Requested {n_queries} queries, but only {n_eligible} match "
                f"question_type={question_type!r}, documents={documents!r}"
            )
        selected_ids = rng.choice(
            eligible_ids,
            size=n_queries,
            replace=False,
        ).tolist()

    logger.info(
        "Selected %d of %d eligible questions for type=%s, documents=%s",
        len(selected_ids),
        n_eligible,
        question_type or "all",
        documents,
    )
    return (
        {qid: queries[qid] for qid in selected_ids},
        {qid: set(relevant_docs[qid]) for qid in selected_ids},
        {qid: query_types[qid] for qid in selected_ids},
        n_eligible,
    )


def add_random_negative_documents(
    source_corpus: dict[str, str],
    relevant_docs: dict[str, set[str]],
    target_size: int,
    rng,
) -> dict[str, str]:
    """Build a corpus containing every positive and random negative documents."""
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    positive_ids = set().union(*relevant_docs.values())
    if not positive_ids.issubset(source_corpus):
        raise ValueError("At least one positive document is missing from the corpus")

    target_size = max(target_size, len(positive_ids))
    if target_size > len(source_corpus):
        raise ValueError(
            f"Requested {target_size} documents from a corpus of "
            f"{len(source_corpus)} documents"
        )

    negative_candidates = sorted(set(source_corpus).difference(positive_ids))
    n_negatives = target_size - len(positive_ids)
    negative_ids = (
        rng.choice(negative_candidates, size=n_negatives, replace=False).tolist()
        if n_negatives
        else []
    )
    selected_ids = sorted(positive_ids) + negative_ids
    corpus = {
        doc_id: source_corpus[doc_id]
        for doc_id in tqdm(selected_ids, desc="Building corpus", unit="doc")
    }
    logger.info(
        "Built corpus with %d positives and %d random negatives",
        len(positive_ids),
        n_negatives,
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
    if set(queries) != set(relevant_docs):
        raise ValueError("Query IDs and relevance IDs differ")
    if any(not document_ids for document_ids in relevant_docs.values()):
        raise ValueError("At least one query has no relevant documents")

    positive_ids = set().union(*relevant_docs.values())
    if len(corpus) != max(expected_corpus_docs, len(positive_ids)):
        raise ValueError(f"Unexpected corpus size: {len(corpus)}")
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
