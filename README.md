# retrieval-benchlab

Experimental benchmark for comparing embedding models, retrieval methods,
similarity functions, and ranking strategies for document retrieval.

The current notebooks build and evaluate a reproducible sample derived from
`BeIR/bioasq-generated-queries`.

## Repository structure

```text
notebooks/
  BioASQ_sample.ipynb    Create, validate, inspect, and save a sample
  Retreaval_test.ipynb   Load the saved sample and evaluate a retriever
src/
  io.py                  Dataset, Google Drive, loading, and persistence helpers
  sampling.py            Query, positive, and negative sampling helpers
```

## Workflow

1. Run [`BioASQ_sample.ipynb`](notebooks/BioASQ_sample.ipynb).
2. Adjust `N_QUERIES`, `N_CORPUS_DOCS`, and `SEED` if needed.
3. The notebook always creates a new sample and replaces
   `Retreaval/data/current` in Google Drive.
4. Inspect a few query and positive-document examples with `EXAMPLE_PAGE`.
5. Run [`Retreaval_test.ipynb`](notebooks/Retreaval_test.ipynb) to evaluate a
   sentence-transformer with cosine similarity.

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

## Dataset caveat

`BeIR/bioasq-generated-queries` contains generated query-document pairs. It is
not identical to the expert-authored BioASQ Task B question set. In the current
sampling procedure every query has exactly one designated positive document.
Results therefore measure retrieval on this generated-query setup and should
not be interpreted as performance on the official multi-document BioASQ gold
annotations.
