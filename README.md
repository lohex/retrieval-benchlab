# retrieval-benchlab

Experimental benchmark for comparing embedding models, lexical and dense
retrievers, similarity functions, and ranking strategies for document retrieval.

The current notebooks build and evaluate reproducible subsets of expert-authored
BioASQ 11b questions. They distinguish question type and the number of relevant
gold documents.

## Repository structure

```text
notebooks/
  BioASQ_sample.ipynb     Create, validate, inspect, and save filtered subsets
  Retreaval_test.ipynb    Register and evaluate retrieval baselines
  Visualize_results.ipynb Analyze datasets and compare registered pipelines
src/
  io.py                   BioASQ metadata, corpus, Drive, and persistence helpers
  sampling.py             Type filters, document filters, and sampling helpers
  dataset_builder.py      Sample construction, persistence, and registration
  retrievers.py           Dense and BM25 ranking backends
  metrics.py              Backend-independent IR metrics
  evaluation_models.py    Pipeline, evaluation, runtime, dataset, and result types
  evaluation_registry.py  Append-only SQLite registry and result persistence
  evaluate.py             Shared retrieval evaluation orchestration
  reporting.py            Read-only dataset and pipeline report tables
TODO.md                    Planned retrieval and full-text extensions
```

## Workflow

1. Run [`BioASQ_sample.ipynb`](notebooks/BioASQ_sample.ipynb).
2. Adjust the sample and calibration configuration if needed.
3. Load the shared benchmark once.
4. Create the common 5,000-document calibration set when testing mean-centered
   dense retrieval.
5. Run the creation blocks for the desired question types.
6. Inspect examples by changing `EXAMPLE_SUBSET` and `EXAMPLE_PAGE`.
7. Run [`Retreaval_test.ipynb`](notebooks/Retreaval_test.ipynb) to register and
   evaluate BM25 and Qwen3-Embedding-0.6B on every latest dataset version.
8. Run [`Visualize_results.ipynb`](notebooks/Visualize_results.ipynb) to inspect
   dataset composition and compare stored pipelines by metric.

The sample notebook supports these six stored subsets:

| Question type | One gold document | Multiple gold documents |
|---|---|---|
| list | `list-one` | `list-multiple` |
| factoid | `factoid-one` | `factoid-multiple` |
| summary | `summary-one` | `summary-multiple` |

Each creation call replaces only its own directory below `Retreaval/data` in
Google Drive. Passing `n_queries=None` uses every eligible question instead of a
fixed-size random sample.

The calibration set is stored separately below
`Retreaval/calibration/bioasq-5k`. It excludes every document annotated as
relevant to a complete `list`, `factoid`, or `summary` question. Its documents
are also excluded from every sampled retrieval corpus.

The notebooks can clone this public repository automatically when they run in
Google Colab:

* [Open the sample notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/BioASQ_sample.ipynb)
* [Open the retrieval notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/Retreaval_test.ipynb)
* [Open the visualization notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/Visualize_results.ipynb)

## Retrieval, evaluation, and runtime identity

Configuration is split according to whether a setting can change rankings,
changes only the reported metric set, or changes only execution behavior.

`PipelineDefinition` contains ranking-relevant settings such as retriever type,
model, similarity function, query instruction, BM25 parameters, and optional
mean-centering data. `EvaluationDefinition` contains only metric names and
cutoffs. `RuntimeConfig` contains batch size, corpus scan size, progress output,
and device.

This means changing GPU, batch size, or scan block size does not create a new
pipeline. Changing NDCG or recall cutoffs creates a new evaluation identity but
not a new retrieval pipeline. Stored result identity is:

```text
pipeline_id + evaluation_id + dataset_id
```

The public API is available from `src.evaluate`:

```python
from src.evaluate import (
    RuntimeConfig,
    evaluate,
    register_dataset,
    register_evaluation,
    register_pipeline,
)
```

## Current baselines

The retrieval notebook currently evaluates two complementary baselines.

BM25 provides a lexical baseline using `rank-bm25` with tokenization kept inside
the repository so its behavior is explicit. Qwen3-Embedding-0.6B provides the
stronger dense baseline and uses an explicit biomedical retrieval instruction on
the query side.

```python
bm25_pipeline_id = register_pipeline(
    retriever_type="bm25",
    registry_db_path=REGISTRY_DB_PATH,
)

qwen_pipeline_id = register_pipeline(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    similarity_metric="cosine",
    query_prompt=(
        "Instruct: Given a biomedical question, retrieve relevant scientific "
        "passages that answer the question\nQuery: "
    ),
    registry_db_path=REGISTRY_DB_PATH,
)
```

Both backends are evaluated by the same in-repository metric implementation.
The default evaluation reports MRR@10, NDCG@10, accuracy and precision/recall at
1, 3, 5, 10, and 100, and MAP@100.

Mean-centered cosine remains supported for dense retrievers through
`embedding_mean`, but it is now an optional ranking configuration rather than
the active default baseline.

## Persistence

Persistent evaluation uses two SQLite databases below `Retreaval/databases` in
Google Drive. `datasets.sqlite` stores immutable pipeline definitions,
evaluation definitions, and versioned dataset identities. New results are stored
in `evaluation_runs_v2` and `metrics_v2` in `results.sqlite`, keyed by pipeline,
evaluation, and dataset. Reporting keeps a fallback for legacy result tables.

Existing results for the same triple are loaded instead of recomputed. Older
registered dataset versions and reverted folder contents are not evaluated.

## Data sources and scope

Question text, question type, and gold PubMed document annotations come from the
official, openly archived
[BioASQ 11b training set](https://zenodo.org/records/7655130). Titles and
abstracts come from the `corpus` configuration of the community dataset
[`DinoStackAI/bioasq-rag-13b-resplit`](https://huggingface.co/datasets/DinoStackAI/bioasq-rag-13b-resplit).

The loader retains only questions for which every annotated gold document is
present in the corpus. Missing documents therefore exclude a complete question
instead of changing its relevance set. Empty source documents are discarded.
Random candidates are sampled from the remaining source corpus after removing
the common calibration documents, so the benchmark does not represent retrieval
against all of PubMed. See [`TODO.md`](TODO.md) for planned hard-candidate and
full-corpus variants.
