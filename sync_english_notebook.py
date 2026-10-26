"""Build the full English report from the executed Russian notebook.

Code, execution counts, and outputs are preserved so both language versions
show the same experiment evidence. Narrative Markdown is translated here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "filmoskop_rag_project.ipynb"
TARGET = ROOT / "filmoskop_rag_project_en.ipynb"

MARKDOWN = {
    0: """# FilmScope Streaming Support RAG Assistant

**Language:** English (default) · [Русская версия](filmoskop_rag_project.ipynb)  
**Repository:** [github.com/enterlina/filmoskop-rag](https://github.com/enterlina/filmoskop-rag)

## Portfolio summary

This project delivers a reproducible, evaluated RAG assistant rather than a
single retrieval demo. The v3 iteration removed 21 archive pages, 13 service
documents, and seven content duplicates without losing any gold-relevant
document. Routed BM25+dense retrieval followed by cross-encoder reranking
reached **recall@5 = 0.9464** and **MRR = 0.8296**. Exact-query recall@5 reached
1.00, semantic 0.96, and multi-document 0.875. GPT-OSS 20B was selected over
Nemotron in the generator A/B test. All expensive stages use atomic,
version-aware checkpoints, and API credentials are never written to disk.

## Project brief

The task is to develop a RAG support assistant for the fictional FilmScope
streaming service. The system must retrieve relevant knowledge-base sections,
answer strictly from retrieved evidence, and abstain when the knowledge base
does not contain the required information.

The source data contains 181 documents and 66 evaluation questions: `exact`
questions with codes, numbers, and product names; conversational `semantic`
paraphrases; `multi` questions requiring several documents; and unanswerable
`gap` questions.

## Objectives

1. Audit the corpus, repair encoding, and remove archive, service, and duplicate content.
2. Build a reproducible retrieval pipeline and reach `recall@5 ≥ 0.85`.
3. Compare dense retrieval, cross-encoder reranking, BM25, Query Expansion, and HyDE.
4. Compare Nemotron and GPT-OSS 20B on faithfulness, relevance, and correctness.
5. Implement consistent abstention and resumable checkpoints.
6. Analyze failures and decide whether the system is ready for a pilot.

## Final architecture

Cleaning and paragraph-aware chunking → multilingual E5 embeddings → routed
BM25+dense RRF for exact requests or dense retrieval otherwise → top-20
candidates → BGE cross-encoder → top-5 contexts → GPT-OSS 20B through the
NVIDIA API.

Components are called directly without LangChain so transformations, rankings,
and metrics remain transparent. Retrieval is evaluated on 56 answerable
questions; ten `gap` questions are evaluated during generation. Run the
notebook from `ypraktikum_task2/filmoskop_project`. The API key is requested
through a hidden prompt and is not stored in the file.
""",
    2: """## Stage 1. Corpus audit and preparation

HTML is parsed, Markdown markers are removed without deleting content, Unicode
is normalized, and mojibake is repaired line by line. Archive and explicit
service pages are filtered by title markers. Near-duplicates are detected from
normalized body content rather than titles, with the newest `doc_id` retained.
The output below provides an auditable list of every exclusion.
""",
    4: """**Stage 1 conclusion.** Cleaning repairs markup and mojibake before
indexing. Empty, archive, and explicit service pages are excluded. Near-exact
duplicates use a 0.92 content-similarity threshold, retaining the newer record.
No gold-relevant document is removed.
""",
    5: """## Stage 2. Chunking and embeddings

Two strategies are compared: whole documents and paragraph-aware chunks up to
1,200 characters. The title is repeated in every paragraph chunk to retain its
subject and context. Splits follow paragraph boundaries rather than cutting
sentences arbitrarily.
""",
    8: """## Stage 3. Retrieval infrastructure and baseline

Exact cosine search is implemented as a matrix product over L2-normalized
vectors, functionally equivalent to FAISS `IndexFlatIP`. For only a few hundred
chunks this is simpler and guarantees exact top-K results. BM25 is routed to
queries containing identifiers, codes, versions, or numbers; RRF combines its
ranking with dense retrieval.
""",
    10: """**Abstention threshold.** A text-only router classifies requests as
exact, semantic, or multi. Each route receives its own fifth-percentile top-1
dense threshold estimated from answerable requests; gap labels are not used to
choose the threshold. Production calibration should use a separate validation
set.
""",
    12: """## Stage 4. Retrieval diagnosis and improvement

Dense search can miss exact codes and rare identifiers, while multi questions
may lose the second required section below top-5. Identifier-heavy queries use
BM25+dense RRF; semantic requests stay on dense search to avoid lexical noise.
The top 20 candidates are passed to a multilingual cross-encoder that scores
the complete query–passage pair.
""",
    14: """The improvement costs a second model and one forward pass over 20
query–passage pairs. CPU inference avoids Apple unified-memory exhaustion but
adds latency, which must be measured on target hardware. The retrieval gain is
large enough to justify reranking in this prototype.
""",
    15: """## Stage 5. Answer generation

Generation uses the NVIDIA OpenAI-compatible API. The key is entered through a
hidden prompt and kept only in kernel memory. Nemotron is retained as the
baseline generator, while GPT-OSS 20B is evaluated as the stronger alternative.
Embeddings, BM25, dense retrieval, and reranking remain local.
""",
    18: """## Optional retrieval experiment: Query Expansion and HyDE

This section tests three short paraphrases and one hypothetical answer document.
Each response is validated as JSON, retried, checkpointed, and replaced by a
marked safe fallback after three malformed API responses. The experiment is
kept separate from the selected retrieval path.
""",
    20: """## Retrieval ablation: Query Expansion and HyDE separately

To avoid attributing a combined result to the wrong component, Query Expansion
and HyDE are evaluated independently with the same reranker and metrics. These
results document a negative experiment: neither method beats the simpler final
pipeline consistently.
""",
    22: """## Stage 6. End-to-end evaluation

The LLM judge returns strict JSON for faithfulness, answer relevance, and
correctness. Gap correctness is also checked deterministically against the
canonical refusal. Oracle evaluation repeats generation with gold documents to
separate retrieval errors from generator errors. Every completed query is
written atomically to a versioned checkpoint.
""",
    24: """## Generator A/B test: GPT-OSS 20B

GPT-OSS is evaluated on exactly the same retrieved contexts as Nemotron and is
stored in a separate checkpoint. This preserves the baseline and makes the
comparison reproducible. Because GPT-OSS is also used as the judge, the report
notes possible self-preference and recommends independent evaluation before
production deployment.
""",
    27: """## Final conclusion and selected configuration

The completed prototype covers corpus audit, line-level mojibake repair,
paragraph-aware chunking, embedding, routed hybrid retrieval, reranking,
grounded generation, abstention, and automated evaluation. Atomic checkpoints
make expensive stages resumable.

### Review iteration

Version 3 removes 21 archive pages, 13 service documents, and seven
content-level near-duplicates without removing any gold-relevant document. The
selected routed hybrid plus cross-encoder reaches `recall@5 = 0.9464` and
`MRR = 0.8296`, compared with `0.7589` and `0.6553` for the clean dense
baseline. Recall@5 is 1.00 for exact, 0.96 for semantic, and 0.875 for multi
questions. No archive page remains in top-five retrieval.

The original A/B test selected GPT-OSS 20B over Nemotron. Because corpus
cleaning changes retrieved contexts, pre-cleaning generator scores are retained
as historical evidence only; v3 generation metrics must be recomputed before a
new end-to-end correctness claim.

### Final parameters

| Component | Selected value |
|---|---|
| Embeddings | `intfloat/multilingual-e5-base` |
| Indexing | Paragraph-aware chunks; normalized embeddings |
| Candidate retrieval | Routed BM25+dense RRF for exact; dense otherwise; top 20 |
| Reranker | `BAAI/bge-reranker-v2-m3`; CPU; batch size 16 |
| Generator context | Top 5 reranked documents |
| Generator | `openai/gpt-oss-20b` through NVIDIA API |
| Generation | `temperature=0`; `max_tokens=2048`; low reasoning effort |
| Judge | `openai/gpt-oss-20b`; JSON output; `temperature=0` |
| Abstention | Canonical refusal; route-specific fifth-percentile thresholds |
| Reproducibility | `seed=42`; atomic JSON/NPZ checkpoints; version 3 |

The retrieval target is exceeded. The v3 generation and judge cells must finish
before comparing correctness with the reviewer target of 0.85. Thresholds
should still be recalibrated on a dedicated validation split before production.
""",
}


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
english = copy.deepcopy(notebook)
for index, text in MARKDOWN.items():
    if english["cells"][index]["cell_type"] != "markdown":
        raise ValueError(f"Expected Markdown cell at index {index}")
    english["cells"][index]["source"] = text.splitlines(keepends=True)

english["metadata"]["language_info"]["name"] = "python"
TARGET.write_text(json.dumps(english, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET.name}: {len(english['cells'])} cells with preserved outputs")
