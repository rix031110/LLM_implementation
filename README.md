# Multilingual RAG Pipeline for Domain-Specific Document Q&A

> Final project — *Text Mining & Data Visualization (AA 2025/26)*
> Group members: Maria Briones, Riccardo Dondi, Sayantan Mandal.

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about a domain-specific corpus. Our primary corpus is the **FATF (Financial Action Task Force) Recommendations on anti-money laundering and countering the financing of terrorism (AML/CFT)**, but the pipeline is lergely corpus-agnostic: we also tested it on an unrelated domain (**Scout manuals**) to confirm it generalizes beyond finance.

This tool is aimed to adress a transversal problem: Specific knowledge is constantly evolving and the need to learn in an interactive way and on short span of time is a growing demand. RAG systems have the advantages of generative IA models while reducing hallucinations. 

---

## 1. Starting point

This project starts from the course repository [nluninja/text-mining-dataviz-aa2526](https://github.com/nluninja/text-mining-dataviz-aa2526),specifically the notebook **`NLP14_1_RAG_Pipeline`**. That notebook gave us a working baseline RAG flow (document store → embeddings → FAISS → a generation model). 

We first adapted the workflow to analyse Scouts manuals (file **RAG.ipynb**) and getting confortable with the different functions and parameters. 

Having the understanding of the workflow, and the uncertanty posted by each stage we rebuilt and extended it into a configurable, reproducible experiment condensensed on **RAG_reorganized.ipynb**. To keep the experiment clean and avoid repeating large cells, the pipeline is written as **reusable functions** parameterized by chunking strategy and generation model. A scenario is then a single function call, which also keeps each run self-contained and reproducible (retrieval resources are passed in explicitly rather than read from globals).

---

## 2. Problem framing

A RAG system has two independent points of failure: **retrieval** (are the right passages found?) and **generation** (does the model produce a faithful answer from those passages?). A weak result can come from either stage, so our work treats them as separate variables and isolates their effects through controlled scenarios rather than tuning everything at once.

We evaluate against **8 factual questions** with known ground-truth answers (e.g. what AML/CFT stands for, when the Recommendations were first drafted and last updated, why non-profit organisations are covered). Short factual answers let us
score correctness with a simple containment check, which keeps the comparison across scenarios objective.

---

## 3. Generation model: the GPT-2 challenge

The baseline used **GPT-2**. We kept it as a control, but it has structural limitations for this task:

- **It is not instruction-tuned.** GPT-2 is a plain causal language model. Even  when the retrieved context clearly contains the answer, it tends to ignore the instruction, continue the prompt, and drift off-topic instead of answering.
- **Short context window (1024 tokens).** Prompt + generation must fit within 1024 tokens. Our augmented prompts (retrieved context + question) are long, so  asking for too many new tokens pushed past the limit and raised an  `IndexError: index out of range in self`. We fixed this by truncating the  input to leave room for generation and by switching from `max_length` to  `max_new_tokens`.
- **Verbose, low-quality answers.** Because it does not "stop" at a clean answer, GPT-2 produced unnecessarily long and often incoherent text.

Because of this, we added **Flan-T5 (`google/flan-t5-base`)** as a second generation model. Flan-T5 is a sequence-to-sequence, instruction-tuned model: its output is *only* the answer (not a continuation of the prompt), and it follows instructions like "answer using only the context below." We expect a clear quality gap in Flan-T5's favour.
GPT-2 is retained as evidence of *why* an instruction-tuned model is needed, not because we expect it to perform well.

---

## 4. Embedding model: limitations and the choice of Qwen

Retrieval quality is bounded by the embedding model. The baseline used `all-MiniLM-L6-v2`, and while investigating retrieval quality we found two limitations that matter for our use case:

- **Context truncation.** `all-MiniLM-L6-v2` has a **256-token** window. Our chunks were ~300 words (~400 tokens), so a meaningful part of each chunk was silently truncated before being embedded.
- **English-centric training.** It is trained primarily on English, which is a   problem for our final goal of testing the same corpus in **three languages**.

### Options we considered

| Embedding model | Context window | Multilingual | Verdict |
|-----------------|---------------|--------------|---------|
| `all-MiniLM-L6-v2` (baseline) | 256 tokens | No (English) | Truncates our chunks; weak for ES/IT |
| `paraphrase-multilingual-MiniLM-L12-v2` / `-mpnet-base-v2` | **128 tokens** (default) | Yes | Multilingual, but truncates **even more**; trained on short texts, so raising the limit does not restore quality |
| **`Qwen/Qwen3-Embedding-0.6B`** | **32K tokens** | Yes (100+ languages) | No truncation of our chunks; genuine multilingual support |

The multilingual `paraphrase-*` models solved the language problem but made the truncation problem *worse* (128-token default, ~80 words), and they were trained on short inputs, so simply raising `max_seq_length` does not recover quality.

### Why Qwen3-Embedding-0.6B

Given that we need **both** multilingual support **and** chunks at least 300 words embedded without truncation, `Qwen3-Embedding-0.6B` was the only option that satisfies both cleanly:

- **32K-token context window** — our chunks are embedded in full, no truncation. The change also allowed us to scale into chunks of 600 words.
- **100+ languages** — supports our English / Spanish / Italian comparison.
- **Instruction-aware queries** — queries are encoded with a task prompt (`prompt_name="query"`) while documents are encoded plainly, matching how the model was trained and improving retrieval.
- Reasonable footprint (~0.6B params) that still runs on a Colab GPU.

The trade-off is cost: it is far larger than MiniLM, needs a GPU, produces 1024-dimensional vectors (vs 384), and is slower to index. For a small corpus this is acceptable, and the retrieval-quality gain justifies it.

---

## 5. Workflow and experimental scenarios

The notebook is organized as a clean six-stage flow:

1. **Set-up** — dependencies and environment.
2. **Data import** — extract text from the source PDF.
3. **Building the Knowledge Base** — chunk the text (one function, three
   strategies: page / paragraph-no-overlap / paragraph-with-overlap).
4. **Retrieval** — embeddings → FAISS index → retrieval function → test.
5. **Augmentation** — build the prompt by injecting retrieved context.
6. **Generation** — produce the answer (GPT-2 or Flan-T5).

We compare **6 scenarios** that cross chunking strategy with generation model:

| # | Chunking strategy | Generation model |
|---|-------------------|------------------|
| 1 | 1 page = 1 document | GPT-2 |
| 2 | paragraph (600 words, no overlap) | GPT-2 |
| 3 | paragraph (600 words, overlap = 50) | GPT-2 |
| 4 | 1 page = 1 document | Flan-T5 |
| 5 | paragraph (600 words, no overlap) | Flan-T5 |
| 6 | paragraph (600 words, overlap = 50) | Flan-T5 |

Each scenario answers the 8 evaluation questions and produces a per-scenario results table (question / expected / generated / match / generation time) plus a summary table of accuracy across scenarios.

### Results summary

<!-- _[ADD SUMMARY TABLE / KEY FINDINGS. Matches per scenario, which chunking strategy and model performed best, and  why.]_ -->

---

## 6. Final mission: multilingual evaluation

<!--The end goal is to take the best-performing scenarios (we expect **Scenario 3** and **Scenario 5**) and compare answers on the **same corpus in three languages — English (current), Spanish, and Italian**. This tests whether retrieval and generation hold up across languages, which is exactly why the embedding model had to be genuinely multilingual.-->

---

## 8. How to run

<!--
_[ADD / ADJUST to match your repo]_

1. Open the notebook in Google Colab (GPU runtime recommended for Qwen).
2. Mount Google Drive and set the path to your source PDF.
3. Run the cells top to bottom: Set-up → Data import → Knowledge Base →
   Retrieval → Augmentation → Generation → Scenarios.

```bash
pip install -r requirements.txt
```

--> 
---

## 9. Repository

We used an incremental commit history and branches throughout the project rather than a single upload at the end.

---

## 10. AI usage

In line with the course AI policy, we disclose our use of AI-based tools.

- **Tool(s) used:** <!--_[e.g. Claude, ChatGPT, Copilot]_.-->
- **Where it contributed:** debugging the generation step (the GPT-2 1024-token `IndexError`), researching and comparing embedding-model context windows and  multilingual support, refactoring the notebook into reusable functions.
- **What remained the group's own work:** the problem framing, the experimental design (the chunking × model scenario matrix), the choice and justification of models, the evaluation methodology, and the interpretation of results.
