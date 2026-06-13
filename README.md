# FATF-RAG — Retrieval-Augmented QA on the FATF Recommendations

> Final project — *Text Mining & Data Visualization (AA 2025/26)*
> Group members: Maria Briones, Riccardo Dondi, Sayantan Mandal.

This is the **Text Mining & Data Visualization** course project for the *RAG on a domain-specific corpus with retrieval evaluation* track. The sections below explain the problem framing, the design choices we made, the alternatives we rejected, and how we interpret our evaluation results.

The provided tool is a domain-specific RAG system that answers questions about the **40 FATF Recommendations (2012)** (*International Standards on Combating Money Laundering and the Financing of Terrorism & Proliferation*), a 126-page document that sets the global AML/CFT standard. 

We chose the FATF Recommendations for their global influence: the text is readily available in multiple languages, which lets us test our model under different linguistic contexts (future work). While we could apply the same principle to a more widely known text such as the Bible, the specificity of the FATF Recommendations makes it less likely that text-generation models were trained on this domain, giving us an exceptional opportunity to detect signs of hallucination.

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

**Answer generation** is fully local with **no API keys**. By default it uses the `hf` backend (open-source Flan-T5 via `transformers`, downloaded on first run). For higher quality, set `LLM_BACKEND=ollama` in `.env`, install [Ollama](https://ollama.com/download), and ollama pull llama3.2`. If the chosen backend isn't available, every entry-point still works and returns cited passages (retrieval-only mode).

## 1. Problem framing

Compliance and related legal frameworks are constantly evolving, in consequence, teams must continously make efforts to stay updated. Since 2012 FAFT has revisited 3 times its 40 reccomendations, this changes in turn encourage modifications on local legal frameworks inside countries. Deveolping domain-specific tools that help working teams to keep up with this evolving framework could mean accelerate workflows and avoid frustration due to over-exposure to training workshops.  

Teams must constantly adress questions like *"Why is enhanced due diligence required?"*, *"What is reccomendation number 6?"*, *"Why implement a risk-based approach?"*. Reading a 126 pages document each time its updated can be slow and most times innecesary. Plain LLM models are not always updated with the sources this specific domain needs. Furthermore, most of the time their answers will not cite the source unless they're explicitly asked for it. LLM's are also prompt to hallucinates plausible-but-wrong specifics, which not acceptable in a regulatory setting.

We framed the task as a **grounded, auditable question-answering**, that is, retrieve the passages that actually contain the answer, and force the model to answer *only* from them, citing the Recommendation and page. Two requirements followed directly from the domain:

1. **Verifiability over fluency.** A confident wrong answer is worse than "I can't find this in the sources." Citations and page-level traceability are first-class features, not decoration.
2. **Robustness to query style.** Users mix conceptual questions (*"what is the risk-based approach?"*) with exact-term lookups (*"Recommendation 16"*, *"USD/EUR 15 000"*). The retriever has to handle both.

## 2. System design

As most compliance teams work on secure environments with wide internet restrictions we opted for a system that runs locally without the external needs of tools like Colab. In consequence, the use of our RAG system is limited by the software and hardware specifications. The use of GPU is required in order for embedding and text generation models to work on acceptable time-frames.

Our **pipeline (`fatf-rag 2/cli.py`)** works as follow:

```
Starting document ──► ingest/chunk ──► embed ──► FAISS index ┐
                          │                     ├─► retriever (dense | bm25 | hybrid) ──► LLM (grounded) ──► cited answer
        BM25 lexical index ─────────────────────┘
```
**Starting document**: a PDF, on this case *40 FATF Recommendations (2012)*.

**Ingestion & chunking (`fatf-rag 2/src/ingest.py`).** We extract page text with `pdfplumber`, strip the recurring header/footer and bare page numbers, then split into chunks (default 250 words / 20 overlap) that prefer to break on paragraph or sentence boundaries. Crucially, **every chunk carries its page span and a best-effort section label** (e.g. *Recommendation 10*, *Interpretive Note to Recommendation 16*, *Glossary*) detected from headings. That metadata is what makes answers citable. 
There are 3 available strategies for chunking that can be compared empirically: `recursive` (character windows with overlap — the production default), `page` (one page per chunk), and `paragraph` (word-based sliding window with optional overlap). The latter two come from our scenario experiments.

**Embeddings (`fatf-rag 2/src/embeddings.py`).** Local `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, L2-normalised. No API key, deterministic, free for the evaluator to reproduce. 

**Vector store (`fatf-rag 2/src/vectorstore.py`).** Exact inner-product search (= cosine on normalised vectors). The default backend is **pure NumPy with no extra dependency**; FAISS (`IndexFlatIP`) is used automatically as an accelerator *if installed* but is entirely optional. An approximate index (IVF/HNSW) would add complexity for no measurable benefit at this scale.

**Retrieval (`fatf-rag 2/src/retriever.py`).** implements a **hybrid** model that min-max normalises each score list over the candidate pool and combines them `alpha*dense + (1-alpha)*bm25` (default `alpha=0.5`).

**Generation (`fatf-rag 2/src/llm.py`).** Decoupled from retrieval and **fully local — no API keys**, with two interchangeable backends:

- **`hf` (default).** Open-source **GPT-2** or **Flan-T5** through Hugging Face  `transformers` (`src/hf_generator.py`). Runs with just `pip install` — nothing else to set up — so the project works end-to-end out of the box. This is also the backend behind our scenario experiments.
- **`ollama`.** A local **Llama** model served by [Ollama](https://ollama.com) at `localhost:11434`, called over its HTTP API using only the Python standard library. Higher answer quality; needs the Ollama app installed.

Both answer *only* from the supplied passages. **If the selected backend isn't available, the system degrades gracefully to retrieval-only mode** with an actionable message instead of failing. Switch backends with `LLM_BACKEND` / `HF_MODEL` in `.env`.

## 3. Alternatives we considered and rejected

- **'Qwen3-Embedding-0.6B' embedding model.** [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) was considered as an alternative to [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), as this last model automatically truncates text longer than 256 words and is mainly trained on english. Nevertheless, Qwen is a considerably heavier model increasing the hardware demand and processing times. Inforcing a larger chunk on all-MiniLM is possible but was discarded as it increased the risk of quality loss, in consequence we opted for chunks of 250 word size.
- **Purely dense or purely lexical (BM25) retrieval.** Each fails where the other succeeds: dense captures meaning and paraphrase but blurs exact terms (acronyms, years, code-like references such as "AML/CFT" or "Recommendation 8"), while BM25 nails exact terms but is blind to synonyms and rephrasing. Their errors are uncorrelated, so a hybrid (fusing both scores) recovers the documents either method alone would miss.

## 4. Evaluation

### Retrieval evaluation

*Run it:*

```bash
python -m eval.evaluate              # current config on the saved index
python -m eval.evaluate --ablation   # retriever + chunk-size comparison
```

Retrieval is evaluated independently of the LLM (the part we control and can measure objectively). We hand-built a **15-question test set** (`eval/testset.json`) and located the **gold page(s)** for each answer manually in the PDF. A retrieved chunk counts as *relevant* if its page span intersects the gold pages. We report:

- **Recall@k** — share of questions with ≥1 relevant chunk in the top-k (here, since each question has a small target set, this equals hit rate / Success@k).
- **MRR** — mean reciprocal rank of the first relevant chunk (does the answer come first?).
- **Precision@k** — average fraction of the top-k that are relevant.

#### Results (verified)

BM25 on the 15-question set (`fatf-rag 2/eval/results/bm25_default.json`):

| Retriever | Recall@5 | MRR | Precision@5 |
|-----------|:-------:|:---:|:-----------:|
| **bm25**  | **1.00** | **0.88** | 0.51 |

Every question retrieves a gold page within the top-5, and on 11/15 the very first result is correct (MRR 0.88). The dense and hybrid rows are produced by the same harness when you run it on a machine with Hugging Face access (the model download is blocked in some sandboxes); the notebook prints all three side by side.

### How we interpret this

- **BM25 is a deliberately strong baseline here, and that is the headline finding, not an anti-climax.** The corpus is small, highly structured, and uses consistent terminology, so questions phrased with the document's own vocabulary are nailed by exact match. A team that reported only a dense-retriever number would be *overstating* the need for embeddings on this corpus.
- **Where dense/hybrid earn their keep is paraphrase** — questions that don't share surface terms with the text (e.g. asking about "money sent abroad" rather than "wire transfer"). Our current test set is mostly in-vocabulary, so it under-tests that gap; see Limitations.
- **Precision@5 ≈ 0.5 is expected and fine.** Several gold answers live on a single page, so at most a couple of the five returned chunks can be "relevant" by our page-intersection rule. Recall and MRR are the metrics that matter for a QA front-end that shows 5 sources.

### Chunking strategy and generation model.

*Run it*: 

```bash
python -m eval.scenarios #(downloads models on first use).
```

Retrieval metrics tell us whether the right text was *found*; they don't tell us whether the generated *answer* is right. We ran an end-to-end experiment (`fatf-rag 2/eval/scenarios.py`, also in `fatf-rag 2/notebooks/rag_scenarios.ipynb`) crossing **three chunking strategies** with **three open-source generators**, scored by whether a short expected answer appears in the output across 8 QA pairs:

| Scenario | Chunking | Model | Correct | Accuracy (%) | 
|----------|----------|-------|---------|--------------|
| 1 | page | GPT-2 | 3 | 37.5 | 
| 2 | paragraph (no overlap) | GPT-2 | 2 | 25.5 |
| 3 | paragraph (overlap=50) | GPT-2 | 2 | 25.5 |
| 4 | page | Flan-T5 | 4 | 50.0 |
| 5 | paragraph (no overlap) | Flan-T5 | 4 | 50.0 |
| 6 | paragraph (overlap=50) | Flan-T5 | 4 | 50.0 |
| 7 | page | llama | 7 | 87.5 |
| 8 | paragraph (no overlap) | llama | 7 | 87.5 |
| 9 | paragraph (overlap=50) | llama | 7 | 87.5 |

*Note:* The accuracy levels displayed here can be found on `fatf-rag 2/eval/results/validated_scenarios.xlsx`, as the csv produced by the code needed expert-validation.

#### What we found and how we read it.

Two effects dominated. First, *chunking strategy doesn't have a vissible meaningfull effect*: the `page` strategy performed reasonably well when compared to small chunking, even giving better result for the case of GPT-2

Second, *the seq2seq model (Flan-T5) beat the causal GPT-2, but both models are beaten by llama*: Flan-T5 is instruction-tuned and extractive-friendly, so it copies the answer out of the context, whereas GPT-2 tends to continue the prompt fluently without actually answering. Llama consistantly provide correct answers and guide its source so it can be tracked back.

## 5. Future work

- Test the model on different languages (ex. Italian and spanish).
- Modify pipeline to integrate more than 1 document as source.
- Explore new embedding models compatible with a loval environment and acceptable time responses.

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

## Working strategy

### Natural intelligence

This project started from the course repository [nluninja/text-mining-dataviz-aa2526](https://github.com/nluninja/text-mining-dataviz-aa2526),specifically the notebook **`NLP14_1_RAG_Pipeline`**. That notebook gave us a working baseline RAG flow (document store → embeddings → FAISS → a generation model). 

We first adapted the workflow to analyse Scouts manuals and getting confortable with the different functions and parameters, we later adapt the workflow to the FAFT analysis.

Having the understanding of the code, and the uncertanty posted by each stage we rebuilt and extended it into a configurable, reproducible experiment condensensed*. To keep the experiment clean and avoid repeating large cells, the pipeline is written as **reusable functions** parameterized by chunking strategy and generation model.

We used an incremental commit history and branches throughout the project rather than a single upload at the end.

### AI usage

Per the course AI policy, here is where AI tools contributed to this project.

- **Tool used:** Claude (Anthropic).
- **Where it helped:** boilerplate for the retrieval pipeline modules (FAISS/BM25 wrappers, the Streamlit and CLI scaffolding), debugging the generation step (the GPT-2 1024-token `IndexError`), researching and comparing embedding-model context windows and  multilingual support, refactoring the notebook into reusable functions and pipeline structure 
- **What remained the group's own work:** the problem framing (grounded, citable QA for a compliance corpus); the design decisions (hybrid retrieval, page-tagged chunking, the choice of open-source local generators, graceful degradation); **the scenario experiment itself** — the chunking strategies, the text-generation model comparison, the 8 QA pairs and the factual-match methodology, originally built by the group in `rag_scenarios.ipynb`; the retrieval evaluation methodology (metric definitions, hand-built test set, gold-page labels); and the interpretation of all results.

The code and text above are our responsibility; any errors are ours.

## License / source

The FATF Recommendations are © 2012 FATF/OECD. The PDF is included here for academic, non-commercial coursework only and is redistributed under the terms of the original publication. This repository's own code is released for educational use.
