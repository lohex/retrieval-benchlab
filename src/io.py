"""Input and output helpers for BioASQ retrieval samples."""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from datasets import Dataset, load_dataset, load_from_disk

from src.evaluation_models import (
    BioASQSample,
    CalibrationSet,
    DatasetValidationError,
)

logger = logging.getLogger(__name__)


def mount_google_drive() -> None:
    """Mount Google Drive when running in Google Colab."""
    try:
        from google.colab import drive
    except ImportError:
        logger.info("Not running in Colab; Google Drive mounting is skipped")
        return

    if not Path("/content/drive/MyDrive").exists():
        logger.info("Mounting Google Drive")
        drive.mount("/content/drive")


def sample_directory(output_root: str | Path, subset_name: str = "current") -> Path:
    """Return the output directory for one named sample or subset."""
    return Path(output_root) / subset_name


def discover_dataset_directories(datasets_root: str | Path) -> list[Path]:
    """Find saved BioASQ sample directories directly below a root folder."""
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


def _read_json(source: str | Path) -> dict[str, Any]:
    """Read JSON from a local path or an HTTP(S) URL."""
    source = str(source)
    if urlparse(source).scheme in {"http", "https"}:
        logger.info("Downloading BioASQ question metadata from %s", source)
        with urlopen(source, timeout=180) as response:
            return json.load(response)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _pubmed_id(document_reference: str) -> str:
    """Extract a PubMed identifier from a BioASQ document URL."""
    return str(document_reference).rstrip("/").rsplit("/", maxsplit=1)[-1]


def load_bioasq_questions(
    questions_source: str | Path,
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, str]]:
    """Load expert-authored questions, types, and gold PubMed documents."""
    payload = _read_json(questions_source)
    questions_data = payload.get("questions")
    if not isinstance(questions_data, list):
        raise ValueError("BioASQ JSON must contain a 'questions' list")

    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    query_types: dict[str, str] = {}

    for question in questions_data:
        qid = str(question["id"])
        body = str(question["body"]).strip()
        question_type = str(question["type"]).lower()
        document_ids = {
            _pubmed_id(reference) for reference in question.get("documents", [])
        }
        if not body:
            raise ValueError(f"Question {qid} has an empty body")
        if not document_ids:
            raise ValueError(f"Question {qid} has no gold documents")

        queries[qid] = body
        relevant_docs[qid] = document_ids
        query_types[qid] = question_type

    if not (set(queries) == set(relevant_docs) == set(query_types)):
        raise ValueError("Question, relevance, and type identifiers differ")

    logger.info(
        "Loaded %d expert questions with type counts %s",
        len(queries),
        dict(sorted(Counter(query_types.values()).items())),
    )
    return queries, relevant_docs, query_types


def load_bioasq_benchmark(
    corpus_dataset_name: str,
    corpus_config: str,
    corpus_split: str,
    questions_source: str | Path,
) -> tuple[
    dict[str, str],
    dict[str, set[str]],
    dict[str, str],
    dict[str, str],
    dict[str, Any],
]:
    """Load BioASQ questions and a corpus containing their PubMed abstracts."""
    from tqdm.auto import tqdm

    logger.info(
        "Loading corpus %s/%s[%s]",
        corpus_dataset_name,
        corpus_config,
        corpus_split,
    )
    dataset = load_dataset(
        corpus_dataset_name,
        corpus_config,
        split=corpus_split,
    )
    required_columns = {"id", "title", "text"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Corpus schema changed; missing columns: {sorted(missing_columns)}"
        )

    corpus: dict[str, str] = {}
    for row in tqdm(dataset, total=len(dataset), desc="Loading corpus", unit="doc"):
        doc_id = str(row["id"])
        title = (row["title"] or "").strip()
        text = (row["text"] or "").strip()
        document = f"{title}\n{text}".strip() if title else text
        if document:
            corpus[doc_id] = document

    queries, relevant_docs, query_types = load_bioasq_questions(questions_source)
    complete_ids = {
        qid
        for qid, document_ids in relevant_docs.items()
        if document_ids.issubset(corpus)
    }
    excluded_ids = set(queries).difference(complete_ids)
    if excluded_ids:
        logger.warning(
            "Excluding %d questions with at least one missing gold document",
            len(excluded_ids),
        )

    queries = {qid: queries[qid] for qid in queries if qid in complete_ids}
    relevant_docs = {
        qid: relevant_docs[qid] for qid in relevant_docs if qid in complete_ids
    }
    query_types = {
        qid: query_types[qid] for qid in query_types if qid in complete_ids
    }
    source_metadata = {
        "corpus_dataset_name": corpus_dataset_name,
        "corpus_config": corpus_config,
        "corpus_split": corpus_split,
        "questions_source": str(questions_source),
        "n_source_questions": len(complete_ids) + len(excluded_ids),
        "n_complete_questions": len(complete_ids),
        "n_excluded_incomplete_questions": len(excluded_ids),
        "n_source_corpus_docs": len(corpus),
        "complete_question_type_counts": dict(
            sorted(Counter(query_types.values()).items())
        ),
    }
    logger.info(
        "Benchmark ready: %d complete questions and %d corpus documents",
        len(queries),
        len(corpus),
    )
    return queries, relevant_docs, query_types, corpus, source_metadata


def load_bioasq_sample(sample_dir: str | Path) -> BioASQSample:
    """Load and validate a sample created by ``BioASQ_sample.ipynb``."""
    sample_dir = Path(sample_dir)
    metadata_path = sample_dir / "metadata.json"
    corpus_path = sample_dir / "corpus"
    logger.info("Loading sample from %s", sample_dir)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    corpus_dataset = load_from_disk(str(corpus_path))
    corpus = dict(zip(corpus_dataset["doc_id"], corpus_dataset["text"]))
    queries = metadata["queries"]
    relevant_docs = {
        qid: set(doc_ids) for qid, doc_ids in metadata["relevant_docs"].items()
    }

    if set(queries) != set(relevant_docs):
        raise ValueError("Query IDs and relevance IDs differ")
    if "query_types" in metadata and set(metadata["query_types"]) != set(queries):
        raise ValueError("Stored query-type IDs differ from query IDs")
    positive_ids = set().union(*relevant_docs.values())
    if not positive_ids.issubset(corpus):
        raise ValueError("At least one positive document is missing")
    if len(queries) != metadata["n_queries"]:
        raise ValueError("Stored query count differs from metadata")
    if len(corpus) != metadata["n_corpus_docs"]:
        raise ValueError("Stored corpus size differs from metadata")

    logger.info(
        "Loaded %d queries, %d positive documents and %d corpus documents",
        len(queries),
        len(positive_ids),
        len(corpus),
    )
    return BioASQSample(
        queries=queries,
        relevant_docs=relevant_docs,
        corpus=corpus,
        metadata=metadata,
    )


def load_calibration_set(calibration_dir: str | Path) -> CalibrationSet:
    """Load and validate a persisted document calibration set."""
    calibration_dir = Path(calibration_dir)
    metadata_path = calibration_dir / "metadata.json"
    corpus_path = calibration_dir / "corpus"
    logger.info("Loading calibration set from %s", calibration_dir)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    corpus_dataset = load_from_disk(str(corpus_path))
    corpus = dict(zip(corpus_dataset["doc_id"], corpus_dataset["text"]))
    expected_documents = metadata.get("n_documents")
    if expected_documents != len(corpus):
        raise ValueError(
            "Stored calibration-set size differs from metadata"
        )
    if any(not document.strip() for document in corpus.values()):
        raise ValueError("At least one calibration document is empty")

    logger.info(
        "Loaded calibration set with %d documents",
        len(corpus),
    )
    return CalibrationSet(corpus=corpus, metadata=metadata)


def _save_document_store(
    output_dir: str | Path,
    corpus: dict[str, str],
    metadata: dict[str, Any],
) -> None:
    """Atomically persist a document corpus and its JSON metadata."""
    output_dir = Path(output_dir)
    staging_dir = output_dir.with_name(f"{output_dir.name}_building")
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Writing new document store to %s", staging_dir)

    Dataset.from_dict(
        {
            "doc_id": list(corpus.keys()),
            "text": list(corpus.values()),
        }
    ).save_to_disk(str(staging_dir / "corpus"))
    (staging_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shutil.rmtree(output_dir, ignore_errors=True)
    staging_dir.replace(output_dir)
    logger.info("New document store saved successfully to %s", output_dir)


def save_sample(
    output_dir: str | Path,
    queries: dict[str, str],
    relevant_docs: dict[str, set[str]],
    corpus: dict[str, str],
    metadata: dict[str, Any],
) -> None:
    """Persist a newly generated sample, replacing the previous one."""
    payload = {
        **metadata,
        "queries": queries,
        "relevant_docs": {
            qid: sorted(doc_ids) for qid, doc_ids in relevant_docs.items()
        },
    }
    _save_document_store(output_dir, corpus, payload)


def save_calibration_set(
    output_dir: str | Path,
    corpus: dict[str, str],
    metadata: dict[str, Any],
) -> None:
    """Persist a newly generated calibration set, replacing the previous one."""
    if metadata.get("n_documents") != len(corpus):
        raise ValueError(
            "Calibration-set size differs from metadata before saving"
        )
    _save_document_store(output_dir, corpus, metadata)
