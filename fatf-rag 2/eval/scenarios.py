"""Answer-level scenario comparison (the group's core experiment).

Reproduces the notebook's 6 scenarios = {chunking strategy} x {generation model}
and scores end-to-end answer quality with a factual substring match over 8 QA
pairs (eval/qa_testset.json):

    | # | chunking              | model   |
    | 1 | page                  | gpt2    |
    | 2 | paragraph no-overlap  | gpt2    |
    | 3 | paragraph overlap=50  | gpt2    |
    | 4 | page                  | flan-t5 |
    | 5 | paragraph no-overlap  | flan-t5 |
    | 6 | paragraph overlap=50  | flan-t5 |

For each scenario: build the knowledge base with that chunking, embed + index
(local MiniLM + cosine), retrieve top-k, build the augmented prompt, generate
with the open-source model, and check whether the expected answer appears.

This needs the HF generation backend (transformers/torch) and downloads gpt2
(~0.5 GB) and flan-t5-base (~1 GB) on first run.

Usage:
    python -m eval.scenarios                 # all 6 scenarios
    python -m eval.scenarios --models flan-t5  # only Flan-T5 scenarios
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import pandas as pd

from src.config import Config
from src.embeddings import Embedder
from src.hf_generator import generate_answer
from src.ingest import build_chunks
from src.llm import create_augmented_prompt
from src.retriever import Retriever
from src.vectorstore import VectorStore

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# (label, chunking strategy, size, overlap) — sizes/overlaps in WORDS for paragraph.
CHUNKINGS = [
    ("page", "page", 0, 0),
    ("paragraph no-overlap", "paragraph", 600, 0),
    ("paragraph overlap=50", "paragraph", 600, 50),
]


def load_qa() -> List[dict]:
    with open(EVAL_DIR / "qa_testset.json") as f:
        return json.load(f)["qa"]


def is_match(expected: str, generated: str) -> bool:
    return expected.lower().strip() in generated.lower().strip()


def _context_block(retrieved) -> str:
    return "\n\n".join(f"[{r.citation}] {r.chunk.text}" for r in retrieved)


def run_scenario(strategy: str, size: int, overlap: int, model_type: str,
                 qa: List[dict], embedder: Embedder, k: int = 2) -> pd.DataFrame:
    chunks = build_chunks(Config().pdf_path, chunk_size=size, chunk_overlap=overlap, strategy=strategy)
    store = VectorStore(chunks, embedder.encode([c.text for c in chunks]))
    retr = Retriever(store, embedder, mode="dense")
    rows = []
    for item in qa:
        retrieved = retr.retrieve(item["question"], top_k=k)
        prompt = create_augmented_prompt(item["question"], _context_block(retrieved))
        t = time.time()
        answer = generate_answer(prompt, model_type)
        rows.append({
            "question": item["question"],
            "expected": item["expected"],
            "generated": answer,
            "match": is_match(item["expected"], answer),
            "gen_time_s": round(time.time() - t, 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "flan-t5"],
                    choices=["gpt2", "flan-t5"])
    ap.add_argument("-k", type=int, default=2)
    args = ap.parse_args()

    qa = load_qa()
    embedder = Embedder(Config().embed_model)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    n = 0
    for model_type in args.models:
        for label, strategy, size, overlap in CHUNKINGS:
            n += 1
            print(f"\n=== Scenario {n}: {label} | {model_type} ===")
            df = run_scenario(strategy, size, overlap, model_type, qa, embedder, k=args.k)
            print(df[["question", "expected", "match"]].to_string(index=False))
            matches = int(df["match"].sum())
            print(f"  -> {matches}/{len(qa)} correct ({100*matches/len(qa):.1f}%)")
            df.to_json(RESULTS_DIR / f"scenario_{model_type}_{strategy}_{overlap}.json",
                       orient="records", indent=2)
            summary_rows.append({"scenario": n, "chunking": label, "model": model_type,
                                 "matches": matches, "accuracy_%": round(100*matches/len(qa), 1)})

    summary = pd.DataFrame(summary_rows)
    print("\n================ SUMMARY ================")
    print(summary.to_string(index=False))
    summary.to_json(RESULTS_DIR / "scenarios_summary.json", orient="records", indent=2)
    print(f"\nSaved -> {RESULTS_DIR/'scenarios_summary.json'}")


if __name__ == "__main__":
    main()
