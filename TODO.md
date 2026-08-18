# TODO

- [ ] Add asymmetric query/document encoder support and a MedCPT baseline. Keep query and document preprocessing explicit in the pipeline identity so dual-encoder retrievers can be compared reproducibly.
- [ ] Add candidate-set difficulty levels: uniform random candidates, BM25-mined hard candidates, and full-corpus retrieval. Treat unjudged BioASQ documents as candidates rather than confirmed negatives.
- [ ] Add HiCBench as a full-text chunking benchmark, including hierarchical gold chunk boundaries and evidence-level retrieval metrics.
