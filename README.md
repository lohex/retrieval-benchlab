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
  Retreaval_test.ipynb   Load one saved subset and evaluate a retriever
src/
  io.py                  BioASQ metadata, corpus, Drive, and persistence helpers
  sampling.py            Type filters, document filters, and sampling helpers
```

## Workflow

1. Run [`BioASQ_sample.ipynb`](notebooks/BioASQ_sample.ipynb).
2. Adjust `N_QUERIES_PER_SUBSET`, `N_CORPUS_DOCS`, and `SEED` if needed.
3. Load the shared benchmark once.
4. Run the creation blocks for the desired question types.
5. Inspect examples by changing `EXAMPLE_SUBSET` and `EXAMPLE_PAGE`.
6. Run [`Retreaval_test.ipynb`](notebooks/Retreaval_test.ipynb), select a
   `SAMPLE_NAME`, and evaluate a sentence-transformer with cosine similarity.

The sample notebook supports these six stored subsets:

| Question type | One gold document | Multiple gold documents |
|---|---|---|
| list | `list-one` | `list-multiple` |
| factoid | `factoid-one` | `factoid-multiple` |
| summary | `summary-one` | `summary-multiple` |

Each creation call replaces only its own directory below
`Retreaval/data` in Google Drive. Passing `n_queries=None` uses every eligible
question instead of a fixed-size random sample.

Both notebooks can clone this public repository automatically when they run in
Google Colab:

* [Open the sample notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/BioASQ_sample.ipynb)
* [Open the retrieval notebook in Colab](https://colab.research.google.com/github/lohex/retrieval-benchlab/blob/main/notebooks/Retreaval_test.ipynb)

## Current evaluation

The retrieval notebook reports:

* NDCG@10
* MRR@10
* Recall@10 and Recall@100
* MAP@100

Cosine similarity is currently configured explicitly as the retrieval score.
The code is structured so that additional embedding models and score functions
can be added without duplicating the data-loading and sampling pipeline.

## Data sources and scope

Question text, question type, and gold PubMed document annotations come from the
official, openly archived
[BioASQ 11b training set](https://zenodo.org/records/7655130). Titles and
abstracts come from the `corpus` configuration of the community dataset
[`DinoStackAI/bioasq-rag-13b-resplit`](https://huggingface.co/datasets/DinoStackAI/bioasq-rag-13b-resplit).

The loader retains only questions for which every annotated gold document is
present in the corpus. Missing documents therefore exclude a complete question
instead of changing its relevance set. Random negatives are sampled from the
44,183-document corpus, so the benchmark does not represent retrieval against
all of PubMed.
