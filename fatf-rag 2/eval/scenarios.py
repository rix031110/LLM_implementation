"""Answer-level scenario comparison (the group's core experiment).

Crosses chunking strategies x generation models and scores end-to-end answer
quality with a factual substring match over the 8 QA pairs in
eval/qa_testset.json.

    chunking strategies : page · paragraph (no overlap) · paragraph (overlap 50)
    generation models   : gpt2 · flan-t5 · llama  (llama requires Ollama running)

For each scenario: build the knowledge base with that chunking, embed + index
(local MiniLM + cosine), retrieve top-k, build the prompt, generate, and check
whether the expected answer appears in the output.

Outputs (in eval/results/):
  * scenarios_review.csv   — one row per (scenario, question) with the FULL
                             generated answer, expected, and the auto match flag,
                             plus a blank `match_manual` column. Open this to read
                             the answers and correct any match the substring test
                             got wrong, then recompute accuracy from your column.
  * scenarios_summary.csv  — matches + accuracy per scenario (auto-computed).
  * scenario_<model>_<strategy>_<overlap>.json — full per-scenario records.

Usage:
    python -m eval.scenarios                       # all available models
    python -m eval.scenarios --models flan-t5 llama
    python -m eval.scenarios --show-chars 400      # longer answer preview in console
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd

from src.config import Config
from src.embeddings import Embedder
from src.ingest import build_chunks
from src.llm import LLM, create_augmented_prompt
from src.retriever import Retriever
from src.vectorstore import VectorStore

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# (label, strategy, size, overlap) — size/overlap in WORDS for the paragraph strategy.
CHUNKINGS = [
    ("page", "page", 0, 0),
    ("paragraph no-overlap", "paragraph", 600, 0),
    ("paragraph overlap=50", "paragraph", 600, 50),
]
ALL_MODELS = ["gpt2", "flan-t5", "llama"]


def load_qa() -> List[dict]:
    with open(EVAL_DIR / "qa_testset.json") as f:
        return json.load(f)["qa"]


def is_match(expected: str, generated: str) -> bool:
    return expected.lower().strip() in generated.lower().strip()


def _context_block(retrieved) -> str:
    return "\n\n".join(f"[{r.citation}] {r.chunk.text}" for r in retrieved)


def make_generator(model_type: str) -> Tuple[Callable[[str, str], str], bool, str]:
    """Return (generate_fn(question, context_block) -> answer, available, note)."""
    if model_type in ("gpt2", "flan-t5"):
        from src import hf_generator
        ok = hf_generator.available()

        def fn(question: str, context: str) -> str:
            prompt = create_augmented_prompt(question, context)
            return hf_generator.generate_answer(prompt, model_type)

        return fn, ok, "transformers (augmented prompt)"

    if model_type == "llama":
        cfg = Config()
        cfg.llm_backend = "ollama"
        cfg.llm_model = "llama3.2"
        llm = LLM(cfg)

        def fn(question: str, context: str) -> str:
            return llm.generate(question, context)

        return fn, llm.available, f"Ollama / {cfg.llm_model} (chat prompt)"

    raise ValueError(f"Unknown model: {model_type}")


def run_scenario(strategy: str, size: int, overlap: int, generate_fn: Callable[[str, str], str],
                 qa: List[dict], embedder: Embedder, k: int = 2) -> pd.DataFrame:
    chunks = build_chunks(Config().pdf_path, chunk_size=size, chunk_overlap=overlap, strategy=strategy)
    store = VectorStore(chunks, embedder.encode([c.text for c in chunks]))
    retr = Retriever(store, embedder, mode="dense")
    rows = []
    for item in qa:
        retrieved = retr.retrieve(item["question"], top_k=k)
        t = time.time()
        answer = generate_fn(item["question"], _context_block(retrieved))
        rows.append({
            "question": item["question"],
            "expected": item["expected"],
            "generated": " ".join(answer.split()),  # collapse whitespace for readability
            "match_auto": is_match(item["expected"], answer),
            "gen_time_s": round(time.time() - t, 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("-k", type=int, default=2)
    ap.add_argument("--show-chars", type=int, default=220, help="answer preview length in console")
    args = ap.parse_args()

    qa = load_qa()
    embedder = Embedder(Config().embed_model)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    review_rows: List[dict] = []
    summary_rows: List[dict] = []
    n = 0
    for model_type in args.models:
        generate_fn, ok, note = make_generator(model_type)
        if not ok:
            print(f"\n!! Skipping '{model_type}' — backend not available ({note}). "
                  + ("Run 'ollama pull llama3.2' and start Ollama." if model_type == "llama"
                     else "Run 'pip install -r requirements.txt'."))
            continue
        for label, strategy, size, overlap in CHUNKINGS:
            n += 1
            print(f"\n=== Scenario {n}: {label} | {model_type}  ({note}) ===")
            df = run_scenario(strategy, size, overlap, generate_fn, qa, embedder, k=args.k)
            for _, r in df.iterrows():
                flag = "match" if r["match_auto"] else "  -  "
                print(f"  [{flag}] Q: {r['question']}")
                print(f"          expected : {r['expected']}")
                print(f"          generated: {r['generated'][:args.show_chars]}")
            matches = int(df["match_auto"].sum())
            print(f"  -> auto {matches}/{len(qa)} match ({100*matches/len(qa):.1f}%)  "
                  "[review scenarios_review.csv to correct any]")
            df.to_json(RESULTS_DIR / f"scenario_{model_type}_{strategy}_{overlap}.json",
                       orient="records", indent=2)
            for _, r in df.iterrows():
                review_rows.append({"scenario": n, "chunking": label, "model": model_type, **r.to_dict()})
            summary_rows.append({"scenario": n, "chunking": label, "model": model_type,
                                 "matches_auto": matches, "accuracy_auto_%": round(100*matches/len(qa), 1)})

    if not summary_rows:
        print("\nNo scenarios ran — no generation backend was available.")
        return

    review = pd.DataFrame(review_rows)
    review["match_manual"] = ""  # blank column: fill TRUE/FALSE after reading the answers
    review_path = RESULTS_DIR / "scenarios_review.csv"
    review.to_csv(review_path, index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "scenarios_summary.csv", index=False)
    summary.to_json(RESULTS_DIR / "scenarios_summary.json", orient="records", indent=2)

    print("\n================ SUMMARY (auto) ================")
    print(summary.to_string(index=False))
    print(f"\nRead answers + correct matches here -> {review_path}")
    print(f"Auto summary -> {RESULTS_DIR/'scenarios_summary.csv'}")


if __name__ == "__main__":
    main()
