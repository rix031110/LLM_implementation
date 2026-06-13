# FATF-RAG — Retrieval-Augmented QA on the FATF Recommendations

A domain-specific RAG system that answers questions about the **FATF Recommendations**
(*International Standards on Combating Money Laundering and the Financing of Terrorism
& Proliferation*, FATF/OECD, Feb 2012 — 126 pages, the global AML/CFT standard). Every
answer is grounded in retrieved passages and cited down to the **Recommendation number
and page**, so it can be verified against the source.

This is the course project for the *RAG on a domain-specific corpus with retrieval
evaluation* track. The sections below explain not just *what* we built but *why* — the
problem framing, the design choices we made, the alternatives we rejected, and how we
read our evaluation results.

---

## 1. Problem framing

Compliance and legal teams repeatedly ask narrow questions of a long, dense regulatory
text: *"When is enhanced due diligence required?"*, *"What must accompany a wire
transfer?"*, *"Who counts as a beneficial owner?"*. Reading 126 pages for each question
is slow, and a plain LLM answers from memory — it cannot cite the source, and it
hallucinates plausible-but-wrong specifics, which is unacceptable in a regulatory
setting.

We framed the task as **grounded, auditable question answering**: retrieve the passages
that actually contain the answer, and force the model to answer *only* from them, citing
the Recommendation and page. Two requirements followed directly from the domain:

1. **Verifiability over fluency.** A confident wrong answer is worse than "I can't find
   this in the sources." Citations and page-level traceability are first-class features,
   not decoration.
2. **Robustness to query style.** Users mix conceptual questions (*"what is the
   risk-based approach?"*) with exact-term lookups (*"Recommendation 16"*, *"USD/EUR
   15 000"*). The retriever has to handle both.

## 2. System design

```
PDF ──► ingest/chunk ──► embed ──► FAISS index ┐
                          │                     ├─► retriever (dense | bm25 | hybrid) ──► LLM (grounded) ──► cited answer
        BM25 lexical index ─────────────────────┘
```

**Ingestion & chunking (`src/ingest.py`).** We extract page text with `pdfplumber`,
strip the recurring header/footer and bare page numbers, then split into overlapping
character chunks (default 900 chars / 150 overlap) that prefer to break on paragraph or
sentence boundaries. Crucially, **every chunk carries its page span and a best-effort
section label** (e.g. *Recommendation 10*, *Interpretive Note to Recommendation 16*,
*Glossary*) detected from headings. That metadata is what makes answers citable.

**Embeddings (`src/embeddings.py`).** Local `sentence-transformers/all-MiniLM-L6-v2`,
384-dim, L2-normalised. No API key, deterministic, free for the evaluator to reproduce.

**Vector store (`src/vectorstore.py`).** Exact inner-product search (= cosine on
normalised vectors). With only ~400 chunks, brute-force search is instant, so the default
backend is **pure NumPy with no extra dependency**; FAISS (`IndexFlatIP`) is used
automatically as an accelerator *if installed* but is entirely optional. An approximate
index (IVF/HNSW) would add complexity for no measurable benefit at this scale.

**Retrieval (`src/retriever.py`).** Three modes: dense (semantic), BM25 (lexical), and a
**hybrid** that min-max normalises each score list over the candidate pool and combines
them `alpha*dense + (1-alpha)*bm25` (default `alpha=0.5`).

**Generation (`src/llm.py`).** Decoupled from retrieval and **fully local — no API keys**,
with two interchangeable backends:

- **`hf` (default).** Open-source **GPT-2** or **Flan-T5** through Hugging Face
  `transformers` (`src/hf_generator.py`). Runs with just `pip install` — nothing else to
  set up — so the project works end-to-end out of the box. This is also the backend behind
  our scenario experiments (see §4).
- **`ollama`.** A local **Llama** model served by [Ollama](https://ollama.com) at
  `localhost:11434`, called over its HTTP API using only the Python standard library.
  Higher answer quality; needs the Ollama app installed.

Both answer *only* from the supplied passages. **If the selected backend isn't available,
the system degrades gracefully to retrieval-only mode** with an actionable message instead
of failing. Switch backends with `LLM_BACKEND` / `HF_MODEL` in `.env`.

**Chunking strategies (`src/ingest.py`).** Three are available so the chunking choice can
be compared empirically: `recursive` (character windows with overlap — the production
default), `page` (one page per chunk), and `paragraph` (word-based sliding window with
optional overlap). The latter two come from our scenario experiments.

## 3. Alternatives we considered and rejected

- **Dense-only retrieval.** Simplest, but it misses exact-term queries. On a small,
  jargon-heavy legal corpus, lexical signal is too valuable to drop — hence hybrid.
- **A managed framework (LangChain / LlamaIndex).** Faster to wire up, but it hides the
  retrieval mechanics we are graded on explaining. We kept the stack thin and transparent
  so every design choice is visible and defensible.
- **LLM-based semantic chunking.** Tempting, but non-deterministic and hard to justify.
  Recursive character splitting with overlap is deterministic, fast, and tunable — and our
  ablation shows it is good enough.
- **A cloud vector DB (Pinecone/Chroma server).** Unnecessary at ~400 chunks; a local
  FAISS file keeps the project fully reproducible with zero infrastructure.
- **A cross-encoder reranker.** A reasonable next step (see §6), but it adds a second
  model and latency. We left it out of the core and flagged it as future work because the
  base retrieval already saturates our test set.

## 4. Evaluation

Retrieval is evaluated independently of the LLM (the part we control and can measure
objectively). We hand-built a **15-question test set** (`eval/testset.json`) and located
the **gold page(s)** for each answer manually in the PDF. A retrieved chunk counts as
*relevant* if its page span intersects the gold pages. We report:

- **Recall@k** — share of questions with ≥1 relevant chunk in the top-k (here, since each
  question has a small target set, this equals hit rate / Success@k).
- **MRR** — mean reciprocal rank of the first relevant chunk (does the answer come first?).
- **Precision@k** — average fraction of the top-k that are relevant.

Run it:

```bash
python -m eval.evaluate              # current config on the saved index
python -m eval.evaluate --ablation   # retriever + chunk-size comparison
```

### Results (verified)

BM25 on the 15-question set (`eval/results/bm25_default.json`):

| Retriever | Recall@5 | MRR | Precision@5 |
|-----------|:-------:|:---:|:-----------:|
| **bm25**  | **1.00** | **0.88** | 0.51 |

Every question retrieves a gold page within the top-5, and on 11/15 the very first result
is correct (MRR 0.88). The dense and hybrid rows are produced by the same harness when you
run it on a machine with Hugging Face access (the model download is blocked in some
sandboxes); the notebook prints all three side by side.

### How we interpret this

- **BM25 is a deliberately strong baseline here, and that is the headline finding, not an
  anti-climax.** The corpus is small, highly structured, and uses consistent terminology,
  so questions phrased with the document's own vocabulary are nailed by exact match. A
  team that reported only a dense-retriever number would be *overstating* the need for
  embeddings on this corpus.
- **Where dense/hybrid earn their keep is paraphrase** — questions that don't share
  surface terms with the text (e.g. asking about "money sent abroad" rather than "wire
  transfer"). Our current test set is mostly in-vocabulary, so it under-tests that gap;
  see Limitations.
- **Precision@5 ≈ 0.5 is expected and fine.** Several gold answers live on a single page,
  so at most a couple of the five returned chunks can be "relevant" by our page-intersection
  rule. Recall and MRR are the metrics that matter for a QA front-end that shows 5 sources.

### Answer-level evaluation: chunking × generation model (scenario experiment)

Retrieval metrics tell us whether the right text was *found*; they don't tell us whether
the generated *answer* is right. So we ran a second, end-to-end experiment (`eval/scenarios.py`,
also in `notebooks/rag_scenarios.ipynb`) crossing **three chunking strategies** with **two
open-source generators**, scored by whether a short expected answer appears in the output
across 8 QA pairs:

| # | Chunking | Model | 
|---|----------|-------|
| 1 | page | GPT-2 |
| 2 | paragraph (no overlap) | GPT-2 |
| 3 | paragraph (overlap=50) | GPT-2 |
| 4 | page | Flan-T5 |
| 5 | paragraph (no overlap) | Flan-T5 |
| 6 | paragraph (overlap=50) | Flan-T5 |

Run it: `python -m eval.scenarios` (downloads GPT-2 and Flan-T5 on first use).

**What we found and how we read it.** Two effects dominated. First, **chunking matters more
than expected**: the `page` strategy performed worst for both models — a whole page is too
much, mostly-irrelevant context, which buries the answer — while the **word-based paragraph
chunks (with a small overlap) gave the best answers**, because the retrieved context is
tight and on-topic. Second, **the seq2seq model (Flan-T5) beat the causal GPT-2**: Flan-T5
is instruction-tuned and extractive-friendly, so it copies the answer out of the context,
whereas vanilla GPT-2 tends to continue the prompt fluently without actually answering. The
takeaway for a compliance assistant is that **retrieval quality and an instruction-following
reader matter more than model size** — a small, well-prompted Flan-T5 over tight paragraph
chunks is a better fit than a larger free-running causal model. (The factual-substring match
is deliberately strict and slightly under-counts paraphrased-but-correct answers, which we
note below.)

## 5. Limitations & honesty

- **The test set is a sanity check, not a benchmark.** 15 questions with page-level,
  hand-assigned gold labels, written by the same group that built the system — small, and
  not independent. It catches regressions and supports the design comparison; it does not
  prove production quality.
- **Page-level relevance is coarse.** A chunk can overlap a gold page without containing
  the precise answer sentence. We accepted this trade-off for labelling speed and
  transparency.
- **The test set is in-vocabulary**, which flatters BM25. A fairer next version would add
  deliberately paraphrased and multi-hop questions.
- **Answer scoring is a strict substring match.** The scenario experiment scores answers
  by whether an expected string appears verbatim, which under-counts paraphrased-but-correct
  answers (we hand-verified a few of these in the notebook). It is a useful comparative
  signal across scenarios, not an absolute accuracy. Faithfulness/citation scoring of
  generated answers is future work.

## 6. Future work

Hybrid `alpha` tuning per query type; a cross-encoder reranker over the top-20; a larger,
paraphrase-heavy and independently-labelled test set; automatic faithfulness/citation
scoring of generated answers; section-aware chunking that never splits a single
Recommendation.

---

## Quickstart

```bash
# 1. Install Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build the index (downloads the embedding model on first run)
python -m scripts.build_index

# 3a. Ask from the command line (generation uses local Flan-T5 by default)
python cli.py "When must enhanced due diligence be applied?"
python cli.py --retriever bm25 --no-generate "Recommendation 16 wire transfer"

# 3b. Or launch the web app
python -m streamlit run app/streamlit_app.py

# 3c. Or open the notebooks
jupyter notebook notebooks/rag_walkthrough.ipynb   # production pipeline
jupyter notebook notebooks/rag_scenarios.ipynb     # chunking x model experiments

# 4. Evaluate
python -m eval.evaluate --ablation   # retrieval metrics + ablation
python -m eval.scenarios             # answer-level chunking x model comparison
```

**Answer generation** is fully local with **no API keys**. By default it uses the `hf`
backend (open-source Flan-T5 via `transformers`, downloaded on first run) — nothing extra
to install. For higher quality, set `LLM_BACKEND=ollama` in `.env`, install
[Ollama](https://ollama.com/download), and `ollama pull llama3.2`. If the chosen backend
isn't available, every entry-point still works and returns cited passages (retrieval-only
mode).

**Offline tests** (no model download, no key) — a quick way to verify the plumbing:

```bash
python -m tests.test_pipeline
```

## Repository layout

```
fatf-rag/
├── data/ENG_REC.pdf            # the corpus
├── src/
│   ├── config.py               # all settings (env-overridable)
│   ├── ingest.py               # PDF -> page-tagged chunks (recursive/page/paragraph)
│   ├── embeddings.py           # sentence-transformers wrapper
│   ├── vectorstore.py          # vector index (NumPy default, FAISS optional)
│   ├── retriever.py            # dense / BM25 / hybrid retrieval
│   ├── llm.py                  # local generation: hf (GPT-2/Flan-T5) | ollama
│   ├── hf_generator.py         # open-source GPT-2 / Flan-T5 generators
│   └── pipeline.py             # end-to-end orchestration (one object, all UIs)
├── scripts/build_index.py      # build/rebuild the index (CLI)
├── cli.py                      # command-line Q&A entry-point
├── app/streamlit_app.py        # web entry-point
├── notebooks/
│   ├── rag_walkthrough.ipynb   # reproducible production-pipeline walkthrough
│   └── rag_scenarios.ipynb     # group's chunking x model scenario experiments
├── eval/
│   ├── testset.json            # hand-built Q&A with gold pages (retrieval)
│   ├── evaluate.py             # Recall@k / MRR / Precision@k + ablation
│   ├── qa_testset.json         # 8 QA pairs (answer-level)
│   ├── scenarios.py            # 6-scenario chunking x model comparison
│   └── results/                # saved metric runs
├── tests/                      # offline smoke tests (hashing embedder)
├── requirements.txt
└── README.md
```

## AI usage

Per the course AI policy, here is where AI tools contributed to this project.

- **Tool used:** Claude (Anthropic).
- **Where it helped:** boilerplate for the retrieval pipeline modules (FAISS/BM25
  wrappers, the Streamlit and CLI scaffolding), drafting docstrings and this README, and
  setting up the evaluation harness structure.
- **What remained the group's own work:** the problem framing (grounded, citable QA for a
  compliance corpus); the design decisions (hybrid retrieval, page-tagged chunking, the
  choice of open-source local generators, graceful degradation); **the scenario experiment
  itself** — the chunking strategies, the GPT-2 vs Flan-T5 comparison, the 8 QA pairs and
  the factual-match methodology, originally built by the group in `rag_scenarios.ipynb`;
  the retrieval evaluation methodology (metric definitions, hand-built test set, gold-page
  labels); and the interpretation of all results — including the finding that chunking and
  an instruction-following reader matter more than model size, that BM25 is a strong
  retrieval baseline here, and the honest account of the test sets' limitations.

The code and text above are our responsibility; any errors are ours.

## License / source

The FATF Recommendations are © 2012 FATF/OECD. The PDF is included here for academic,
non-commercial coursework only and is redistributed under the terms of the original
publication. This repository's own code is released for educational use.
