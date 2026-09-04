# retrieval-benchlab

Experimental benchmark for comparing embedding models, lexical and dense
retrievers, similarity functions, and ranking strategies for document retrieval.

The current notebooks build and evaluate reproducible subsets of expert-authored
BioASQ 11b questions. They distinguish question type and the number of relevant
gold documents.

## Repository structure

```text
notebooks/
  BioASQ_sample.ipynb              Create, validate, inspect, and save filtered subsets
  Retrieval_baselines.ipynb        Compare BM25, MiniLM, and Qwen3 baselines
  MiniLM_embedding_ablations.ipynb Compare calibrated and query-adapted MiniLM scoring
  Visualize_results.ipynb          Analyze datasets and compare registered pipelines
src/
  io.py                            BioASQ metadata, corpus, Drive, and persistence helpers
  sampling.py                      Type filters, document filters, and sampling helpers
  dataset_builder.py               Sample construction, persistence, and registration
  embedding_transforms.py          Dense embedding transformation configuration
  retrievers.py                    Dense and BM25 ranking backends
  metrics.py                       Backend-independent IR metrics
  evaluation_models.py             Pipeline, evaluation, runtime, dataset, and result types
  evaluation_registry.py           Append-only SQLite registry and result persistence
  evaluate.py                      Shared retrieval evaluation orchestration
  reporting.py                     Read-only dataset and pipeline report tables
TODO.md                             Planned retrieval and full-text extensions
```

## Workflow

1. Run [`BioASQ_sample.ipynb`](notebooks/BioASQ_sample.ipynb).
2. Adjust the sample and calibration configuration if needed.
3. Load the shared benchmark once.
4. Create the common 5,000-document calibration set when testing calibrated
   dense embedding transformations.
5. Run the creation blocks for the desired question types.
6. Inspect examples by changing `EXAMPLE_SUBSET` and `EXAMPLE_PAGE`.
7. Run [`Retrieval_baselines.ipynb`](notebooks/Retrieval_baselines.ipynb) for the
   unmodified BM25, MiniLM, and Qwen3 comparison.
8. Run [`MiniLM_embedding_ablations.ipynb`](notebooks/MiniLM_embedding_ablations.ipynb)
   for MiniLM mean-centering, variance normalization, z-normalization, and
   query-adapted weighted cosine.
9. Run [`Visualize_results.ipynb`](notebooks/Visualize_results.ipynb) to inspect
   dataset composition and compare stored pipelines by metric.

The sample notebook supports six stored subsets:

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

## Open in Google Colab

The notebooks clone the current `main` branch automatically, install their required Python packages, and use the configured Google Drive paths for datasets and SQLite databases.

* [Open the baseline comparison notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/Retrieval_baselines.ipynb)
* [Open the MiniLM embedding ablation notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/MiniLM_embedding_ablations.ipynb)
* [Open the sample notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/BioASQ_sample.ipynb)
* [Open the visualization notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/Visualize_results.ipynb)

## Retrieval, evaluation, and runtime identity

Configuration is split according to whether a setting can change rankings,
changes only the reported metric set, or changes only execution behavior.

`PipelineDefinition` contains ranking-relevant settings such as retriever type,
model, similarity function, query instruction, BM25 parameters, and dense
embedding transformation/scoring configuration. `EvaluationDefinition` contains
only metric names and cutoffs. `RuntimeConfig` contains batch size, corpus scan
size, progress output, and device.

Dense retrieval follows an explicit path:

```text
raw embeddings -> embedding transform -> similarity/scoring
```

The identity transform leaves embeddings unchanged. Calibrated transforms
include mean centering, variance normalization, and z-normalization. Calibration
statistics are estimated dimension-wise from raw, non-L2-normalized calibration
document embeddings. `mu`, `sigma`, calibration `source_id`, transform type, and
numerical parameters are stored in the ranking-relevant pipeline configuration.

Query-adapted MiniLM scoring starts from z-normalized embeddings and uses
`w_k = |z_q,k|^alpha` inside weighted cosine. `alpha` is part of pipeline
identity. `alpha = 0` reduces to standard cosine in the z-normalized space.

Changing GPU, batch size, or scan block size does not create a new pipeline.
Changing the embedding transform, calibration statistics, query-adaptation
parameter, model, similarity metric, or query instruction does. Changing metric
cutoffs creates a new evaluation identity but not a new retrieval pipeline.
Stored result identity is:

```text
pipeline_id + evaluation_id + dataset_id
```

The public API is available from `src.evaluate`:

```python
from src.evaluate import (
    RuntimeConfig,
    compute_calibration_statistics,
    evaluate,
    register_dataset,
    register_evaluation,
    register_pipeline,
)
```

## Current baseline comparison

`Retrieval_baselines.ipynb` compares three complementary baselines without
embedding post-processing:

- BM25
- `sentence-transformers/all-MiniLM-L6-v2` with cosine similarity
- `Qwen/Qwen3-Embedding-0.6B` with cosine similarity and a biomedical query instruction

All backends use the same in-repository metric implementation. The default
evaluation reports MRR@10, NDCG@10, accuracy and precision/recall at 1, 3, 5,
10, and 100, and MAP@100.

## Persistence

Persistent evaluation uses two SQLite databases below `Retreaval/databases` in
Google Drive. `datasets.sqlite` stores immutable pipeline definitions,
evaluation definitions, and versioned dataset identities. `results.sqlite`
stores `evaluation_runs` and `metrics`, keyed by pipeline, evaluation, and
dataset.

Existing results for the same triple are loaded instead of recomputed. Older
registered dataset versions and reverted folder contents are not evaluated. The
final cell of `Visualize_results.ipynb` can optionally delete both databases for
a clean reset; `RESET_DATABASES` is `False` by default.

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
