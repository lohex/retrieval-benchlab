# retrieval-benchlab

Experimental benchmark for comparing embedding models, retrieval methods,
similarity functions, and ranking strategies for document retrieval.

The current notebooks build and evaluate reproducible subsets of expert-authored
BioASQ 11b questions. They distinguish question type and the number of relevant
gold documents.

## Repository structure

```text
notebooks/
  BioASQ_sample.ipynb    Create, validate, inspect, and save filtered subsets
  Retreaval_test.ipynb   Register and evaluate a pipeline on all datasets
  Visualize_results.ipynb Analyze datasets and compare registered pipelines
src/
  io.py                  BioASQ metadata, corpus, Drive, and persistence helpers
  sampling.py            Type filters, document filters, and sampling helpers
  dataset_builder.py     Sample construction, persistence, and registration
  evaluation_models.py   Typed samples, pipeline definitions, and outcomes
  evaluation_registry.py Append-only SQLite registry and result persistence
  evaluate.py             Sentence-transformer evaluation orchestration
  reporting.py            Read-only dataset and pipeline report tables
```

## Workflow

1. Run [`BioASQ_sample.ipynb`](notebooks/BioASQ_sample.ipynb).
2. Adjust the sample and calibration configuration if needed.
3. Load the shared benchmark once.
4. Create the common 5,000-document calibration set.
5. Run the creation blocks for the desired question types.
6. Inspect examples by changing `EXAMPLE_SUBSET` and `EXAMPLE_PAGE`.
7. Run [`Retreaval_test.ipynb`](notebooks/Retreaval_test.ipynb), register a
   pipeline, and evaluate it on every latest dataset version.
8. Run [`Visualize_results.ipynb`](notebooks/Visualize_results.ipynb) to inspect
   dataset composition and compare all stored pipelines by metric.

The sample notebook supports these six stored subsets:

| Question type | One gold document | Multiple gold documents |
|---|---|---|
| list | `list-one` | `list-multiple` |
| factoid | `factoid-one` | `factoid-multiple` |
| summary | `summary-one` | `summary-multiple` |

Each creation call replaces only its own directory below
`Retreaval/data` in Google Drive. Passing `n_queries=None` uses every eligible
question instead of a fixed-size random sample.

The calibration set is stored separately below
`Retreaval/calibration/bioasq-5k`. It excludes every document annotated as
relevant to a complete `list`, `factoid`, or `summary` question. Its documents
are also excluded from every sampled retrieval corpus.

Both notebooks can clone this public repository automatically when they run in
Google Colab:

* [Open the sample notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/BioASQ_sample.ipynb)
* [Open the retrieval notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/Retreaval_test.ipynb)
* [Open the visualization notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/Visualize_results.ipynb)

## Current evaluation

The retrieval notebook reports:

* NDCG@10
* MRR@10
* Recall@10 and Recall@100
* MAP@100

Cosine similarity is currently configured explicitly as the retrieval score.
The code is structured so that additional embedding models and score functions
can be added without duplicating the data-loading and sampling pipeline.

Persistent evaluation uses two SQLite databases below
`Retreaval/databases` in Google Drive. `datasets.sqlite` stores immutable
pipeline definitions and versioned dataset identities. `results.sqlite` stores
one result for every `pipeline_id` and `dataset_id` combination. The public API
remains available from `src.evaluate`:

```python
from src.evaluate import evaluate, register_dataset, register_pipeline
```

The test notebook calls `evaluate(pipeline_id, datasets_root)`. Existing results
are loaded from `results.sqlite`, while missing combinations are evaluated and
appended. Older registered versions and reverted folder contents are not
evaluated.

## Data sources and scope

Question text, question type, and gold PubMed document annotations come from the
official, openly archived
[BioASQ 11b training set](https://zenodo.org/records/7655130). Titles and
abstracts come from the `corpus` configuration of the community dataset
[`DinoStackAI/bioasq-rag-13b-resplit`](https://huggingface.co/datasets/DinoStackAI/bioasq-rag-13b-resplit).

The loader retains only questions for which every annotated gold document is
present in the corpus. Missing documents therefore exclude a complete question
instead of changing its relevance set. Empty source documents are discarded.
Random negatives are sampled from the remaining source corpus after removing
the common calibration documents, so the benchmark does not represent
retrieval against all of PubMed.
