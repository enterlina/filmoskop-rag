# FilmScope RAG Assistant

An evaluated Russian-language support RAG system built as an AI Engineering
portfolio project.

**Default report:** [English version](filmoskop_rag_portfolio_en.ipynb)  
**Also available:** [Русская версия](filmoskop_rag_project.ipynb)

The final v3 pipeline performs corpus cleaning, paragraph-aware chunking,
multilingual E5 retrieval, routed BM25+dense RRF for exact identifiers,
cross-encoder reranking, grounded GPT-OSS generation, abstention, and
checkpointed evaluation.

## Final retrieval results

| Configuration | Recall@5 | MRR |
|---|---:|---:|
| Clean dense baseline | 0.7589 | 0.6553 |
| Routed BM25 + dense | 0.7679 | 0.6529 |
| Hybrid + cross-encoder | **0.9464** | **0.8296** |

The review iteration removed 21 archive pages, 13 service documents, and seven
content-level near-duplicates without removing any gold-relevant document.

## Project files

- [`filmoskop_rag_portfolio_en.ipynb`](filmoskop_rag_portfolio_en.ipynb) —
  default English portfolio report;
- [`filmoskop_rag_project.ipynb`](filmoskop_rag_project.ipynb) — complete
  Russian report and implementation;
- [`README-project.md`](README-project.md) — detailed setup and engineering
  notes;
- [`filmoskop_checkpoints/retrieval_v3_results.json`](filmoskop_checkpoints/retrieval_v3_results.json)
  — machine-readable final retrieval metrics.

## Run locally

```bash
python -m pip install -r requirements-project.txt
jupyter notebook filmoskop_rag_project.ipynb
```

The dataset downloads automatically from the course source. The NVIDIA API key
is requested through a hidden prompt and is never stored in the notebooks.

