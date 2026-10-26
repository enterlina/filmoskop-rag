# FilmScope RAG Assistant

**Repository:** [github.com/enterlina/filmoskop-rag](https://github.com/enterlina/filmoskop-rag)

An end-to-end Russian-language RAG prototype for a fictional streaming-service
support knowledge base. The project covers data cleaning, mojibake repair,
paragraph-aware chunking, dense retrieval, cross-encoder reranking, grounded
answer generation, abstention, checkpointing, and LLM-based evaluation.

## Results

The selected retrieval pipeline is multilingual E5 plus routed BM25/RRF for
exact identifiers → top-20 candidates → BGE cross-encoder reranking → top-5
contexts. Before indexing, archive/service pages and content-level duplicates
are removed with an auditable exclusion table.

| Metric | Clean dense baseline | Selected hybrid + reranker |
|---|---:|---:|
| Recall@5 | 0.7589 | **0.9464** |
| MRR | 0.6553 | **0.8296** |

GPT-OSS 20B was selected for generation after outperforming Nemotron 3.5
Lightning in the original A/B test. Because corpus cleaning changes the
retrieved contexts, v3 generation metrics must be recomputed and are stored in
a separate versioned checkpoint rather than mixed with the earlier scores.

Query Expansion, HyDE, and their combination were evaluated separately. None
outperformed the simpler reranked pipeline, so they remain ablation experiments
rather than production-path components.

## Notebooks

- `filmoskop_rag_project.ipynb` — full Russian-language implementation and
  experiments.
- `filmoskop_rag_portfolio_en.ipynb` — concise English portfolio narrative,
  comparisons, engineering decisions, and final configuration.

## Setup

```bash
python -m pip install -r requirements-project.txt
jupyter notebook filmoskop_rag_project.ipynb
```

The retrieval stack uses:

- `intfloat/multilingual-e5-base` for normalized embeddings;
- `BAAI/bge-reranker-v2-m3` for CPU reranking;
- `openai/gpt-oss-20b` through the NVIDIA OpenAI-compatible API for generation
  and evaluation.

The notebook requests the NVIDIA API key through a hidden prompt. Credentials
are not stored in either notebook or checkpoint files.

## Reproducibility

Expensive stages are saved as atomic JSON/NPZ checkpoints. Cache keys contain
model names, settings, input data, and the preprocessing version, preventing
results from incompatible experiments from being mixed. The random seed is 42.

Cross-encoder inference intentionally runs on CPU to avoid shared Metal-memory
pressure on macOS. The prototype calibrates its abstention threshold from the
available answerable questions; a production system should use a separate
validation split.

## Status

The retrieval target of recall@5 ≥ 0.85 was exceeded. End-to-end correctness
must be recomputed for the v3 corpus before making a final production-readiness
claim; the project remains a research/educational prototype.
