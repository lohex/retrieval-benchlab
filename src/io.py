"""Input and output helpers for BioASQ retrieval samples."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
    load_from_disk,
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


def sample_directory(output_root: str | Path) -> Path:
    """Return the directory used for the latest generated sample."""
    return Path(output_root) / "current"


def inspect_and_load_dataset(
    dataset_name: str,
    config: str,
    split: str,
):
    """Inspect the source schema and load the requested Hugging Face split."""
    logger.info("Inspecting dataset configuration")
    configs = get_dataset_config_names(dataset_name)
    splits = get_dataset_split_names(dataset_name, config)
    logger.info("Available configs: %s", configs)
    logger.info("Available splits: %s", splits)

    dataset = load_dataset(dataset_name, config, split=split)
    required_columns = {"_id", "title", "text", "query"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset schema changed; missing columns: {sorted(missing_columns)}"
        )

    logger.info("Rows: %d", len(dataset))
    logger.info("Columns: %s", dataset.column_names)
    return dataset


def load_bioasq_sample(
    sample_dir: str | Path,
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, str], dict[str, Any]]:
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
    return queries, relevant_docs, corpus, metadata


def save_sample(
    output_dir: str | Path,
    queries: dict[str, str],
    relevant_docs: dict[str, set[str]],
    corpus: dict[str, str],
    metadata: dict[str, Any],
) -> None:
    """Persist a newly generated sample, replacing the previous one."""
    output_dir = Path(output_dir)
    staging_dir = output_dir.with_name(f"{output_dir.name}_building")
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Writing new sample to staging directory %s", staging_dir)

    Dataset.from_dict(
        {
            "doc_id": list(corpus.keys()),
            "text": list(corpus.values()),
        }
    ).save_to_disk(str(staging_dir / "corpus"))

    payload = {
        **metadata,
        "queries": queries,
        "relevant_docs": {
            qid: sorted(doc_ids) for qid, doc_ids in relevant_docs.items()
        },
    }
    (staging_dir / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shutil.rmtree(output_dir, ignore_errors=True)
    staging_dir.replace(output_dir)
    logger.info("New sample saved successfully to %s", output_dir)
