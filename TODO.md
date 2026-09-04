# TODO

## 1. Refactor embedding transformations before splitting notebooks

- [x] Introduce an explicit embedding-transformation layer between dense encoding and similarity scoring:
  - `raw embeddings -> embedding transform -> similarity`
  - Keep `DenseRetriever` responsible for encoding and ranking rather than accumulating transformation-specific branches.
  - Represent the selected transformation and all ranking-relevant transformation parameters in `PipelineDefinition` so pipeline identity changes whenever the scoring geometry changes.
- [x] Generalize calibration support from a mean-only helper to reusable per-dimension calibration statistics:
  - estimate `mu` and `sigma` from raw, non-L2-normalized document embeddings of a fixed calibration corpus that is disjoint from evaluation datasets;
  - use an epsilon floor for very small `sigma` values;
  - make the calibration source/identity and transformation parameters reproducible and ranking-relevant.
- [x] Keep standard cosine as the identity/no-op transform so the same dense-retrieval path supports both unmodified baselines and later embedding-space ablations.

## 2. Split the evaluation workflow into two notebooks

### Notebook 1: baseline model comparison

- [x] Create `notebooks/Retrieval_baselines.ipynb` comparing the three retrieval baselines directly, without mean subtraction or other embedding-space post-processing:
  - **BM25** as the lexical baseline.
  - **`sentence-transformers/all-MiniLM-L6-v2`** as the original dense baseline, using standard cosine similarity.
  - **`Qwen/Qwen3-Embedding-0.6B`** as the modern dense baseline, using cosine similarity and the explicit biomedical query instruction while documents are encoded without that instruction.
- [x] Keep the evaluation definition and dataset set identical across all three baselines so the comparison isolates the retrieval model/backend.
- [x] Report the same stored IR metrics for every baseline and keep runtime settings outside pipeline identity.

### Notebook 2: MiniLM embedding-space ablations

Use **`sentence-transformers/all-MiniLM-L6-v2`** only. The goal is to test the hypothesis that many embedding dimensions contribute mostly non-query-specific background or noise, while dimensions in which the query is unusual relative to the document distribution should receive more weight.

Use the calibration infrastructure from step 1. Encode each calibration document into one sentence/document embedding without L2 normalization, then estimate dimension-wise statistics across document embeddings.

- [x] Create `notebooks/MiniLM_embedding_ablations.ipynb`.
- [x] **Standard cosine** as the reference condition.
- [x] **Mean subtraction / mean-centered cosine**:
  - estimate a dimension-wise document mean `mu` from the calibration embeddings;
  - transform both query and document embeddings with `x' = x - mu`;
  - L2-normalize only after centering, then compute cosine similarity.
- [x] **Variance-normalized cosine**:
  - estimate the dimension-wise standard deviation `sigma` on calibration document embeddings;
  - scale with `x' = x / max(sigma, epsilon)`.
- [x] **Z-normalized cosine**:
  - use `z = (x - mu) / max(sigma, epsilon)` for both queries and documents;
  - L2-normalize the resulting vectors before cosine similarity.
- [x] **Query-adapted weighted cosine**:
  - start from z-normalized query and document embeddings;
  - derive query-specific weights `w_k = |z_q,k|^alpha`;
  - use weighted cosine so dimensions in which the query is unusually far from the corpus distribution receive more influence;
  - store `alpha` as part of the ranking-relevant transform configuration;
  - `alpha = 0` is mathematically equivalent to the z-normalized cosine reference, and the notebook uses `alpha = 1` as the first adapted condition.
- [x] Compare all implemented variants on exactly the same datasets and evaluation definition as Notebook 1.
- [ ] Add **top-k dimension gating** as a follow-up experiment: rank dimensions by `|z_q,k|` for each query and retain only the most query-specific fraction or number of dimensions before scoring. Test at least 25%, 50%, and 75%; a broader sweep of 5%, 10%, 25%, 50%, 75%, and 100% is preferred.

## Retrieval extensions

- [ ] Add asymmetric query/document encoder support and a MedCPT baseline. Keep query and document preprocessing explicit in the pipeline identity so dual-encoder retrievers can be compared reproducibly.
- [ ] Add candidate-set difficulty levels: uniform random candidates, BM25-mined hard candidates, and full-corpus retrieval. Treat unjudged BioASQ documents as candidates rather than confirmed negatives.
- [ ] Add HiCBench as a full-text chunking benchmark, including hierarchical gold chunk boundaries and evidence-level retrieval metrics.
