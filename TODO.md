# TODO

## 1. Refactor embedding transformations before splitting notebooks

- [ ] Introduce an explicit embedding-transformation layer between dense encoding and similarity scoring:
  - `raw embeddings -> embedding transform -> similarity`
  - Keep `DenseRetriever` responsible for encoding and ranking rather than accumulating transformation-specific branches.
  - Represent the selected transformation and all ranking-relevant transformation parameters in `PipelineDefinition` so pipeline identity changes whenever the scoring geometry changes.
- [ ] Generalize calibration support from a mean-only helper to reusable per-dimension calibration statistics:
  - estimate `mu` and `sigma` from raw, non-L2-normalized document embeddings of a fixed calibration corpus that is disjoint from evaluation datasets;
  - use an epsilon floor for very small `sigma` values;
  - make the calibration source/identity and transformation parameters reproducible and ranking-relevant.
- [ ] Keep standard cosine as the identity/no-op transform so the same dense-retrieval path supports both unmodified baselines and later embedding-space ablations.

## 2. Split the evaluation workflow into two notebooks

### Notebook 1: baseline model comparison

- [ ] Create a notebook that compares the three retrieval baselines directly, without mean subtraction or any other embedding-space post-processing:
  - **BM25** as the lexical baseline.
  - **`sentence-transformers/all-MiniLM-L6-v2`** as the original dense baseline, using its standard cosine similarity.
  - **`Qwen/Qwen3-Embedding-0.6B`** as the modern dense baseline, using cosine similarity and the explicit biomedical query instruction while documents are encoded without that instruction.
- [ ] Keep the evaluation definition and dataset set identical across all three baselines so the comparison isolates the retrieval model/backend.
- [ ] Report the same stored IR metrics for every baseline and keep runtime settings outside pipeline identity.

### Notebook 2: MiniLM embedding-space ablations

Use **`sentence-transformers/all-MiniLM-L6-v2`** only. The goal is to test the hypothesis that many embedding dimensions contribute mostly non-query-specific background or noise, while dimensions in which the query is unusual relative to the document distribution should receive more weight.

Use the calibration infrastructure from step 1. Encode each calibration document into one sentence/document embedding without L2 normalization, then estimate dimension-wise statistics across document embeddings.

- [ ] **Standard cosine** as the reference condition.
- [ ] **Mean subtraction / mean-centered cosine**:
  - Estimate a dimension-wise document mean `mu` from the calibration embeddings.
  - Transform both query and document embeddings with `x' = x - mu`.
  - L2-normalize only after centering, then compute cosine similarity.
  - Motivation: dimensions where the query is close to the typical corpus value are suppressed, whereas dimensions where the query deviates strongly from the corpus mean contribute more strongly.
- [ ] **Variance-normalized cosine**:
  - Estimate the dimension-wise standard deviation `sigma` on the calibration document embeddings.
  - Scale dimensions by their typical variability, e.g. `x' = x / sigma`, with an epsilon floor for numerically small `sigma`.
  - This tests whether dimensions with intrinsically high variance dominate similarity even when that variance is not query-specific.
- [ ] **Z-normalized cosine**:
  - Use `z = (x - mu) / sigma` for both queries and documents, again with an epsilon floor.
  - L2-normalize the resulting vectors before cosine similarity.
  - This combines corpus centering with dimension-wise variance normalization and is equivalent to a diagonal whitening approximation.
- [ ] **Query-adapted weighted cosine**:
  - Start from the z-normalized query and document embeddings.
  - Derive query-specific dimension weights from the magnitude of the query deviation, for example `w_k = |z_q,k|^alpha`.
  - Compute a weighted cosine similarity so dimensions in which the query is unusually far from the corpus distribution receive more influence.
  - Treat `alpha` as an explicit pipeline parameter and include at least `alpha = 0` as the z-normalized reference and one or more values `> 0` for stronger query adaptation.
- [ ] Compare all variants on exactly the same datasets and evaluation definition as Notebook 1.
- [ ] Add **top-k dimension gating** as a follow-up experiment: rank dimensions by `|z_q,k|` for each query and retain only the most query-specific fraction or number of dimensions before scoring. Test several gates, for example 25%, 50%, and 75%, to directly test the hypothesis that many embedding dimensions mainly add noise.

## Retrieval extensions

- [ ] Add asymmetric query/document encoder support and a MedCPT baseline. Keep query and document preprocessing explicit in the pipeline identity so dual-encoder retrievers can be compared reproducibly.
- [ ] Add candidate-set difficulty levels: uniform random candidates, BM25-mined hard candidates, and full-corpus retrieval. Treat unjudged BioASQ documents as candidates rather than confirmed negatives.
- [ ] Add HiCBench as a full-text chunking benchmark, including hierarchical gold chunk boundaries and evidence-level retrieval metrics.
