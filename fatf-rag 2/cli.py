"""Command-line entry-point for querying the FATF-RAG system.

Examples
--------
    python cli.py "What is a beneficial owner?"
    python cli.py --top-k 8 --retriever bm25 "When is enhanced due diligence required?"
    python cli.py --no-generate "Recommendation 16 wire transfer requirements"   # retrieval only
    python cli.py                      # interactive REPL
"""
from __future__ import annotations

import argparse
import sys

from src.config import Config
from src.pipeline import RAGPipeline


def _print_response(resp) -> None:
    print("\n" + "=" * 70)
    print("ANSWER\n" + "-" * 70)
    print(resp.answer)
    print("\nSOURCES\n" + "-" * 70)
    print(resp.format_sources())
    print("=" * 70 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the FATF Recommendations via RAG.")
    ap.add_argument("question", nargs="*", help="Question text. Omit for interactive mode.")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default=None)
    ap.add_argument("--no-generate", action="store_true", help="Retrieval only, skip the LLM.")
    args = ap.parse_args()

    cfg = Config()
    if args.retriever:
        cfg.retriever = args.retriever
    if args.top_k:
        cfg.top_k = args.top_k

    try:
        pipe = RAGPipeline.from_storage(cfg)
    except FileNotFoundError:
        print("No index found. Run:  python -m scripts.build_index", file=sys.stderr)
        sys.exit(1)

    if not pipe.llm.available and not args.no_generate:
        print(f"[info] Generation backend '{pipe.llm.backend}' unavailable — retrieval-only "
              "mode. For 'hf' run 'pip install -r requirements.txt'; for 'ollama' install "
              "Ollama and 'ollama pull llama3.2'.\n")

    if args.question:
        resp = pipe.query(" ".join(args.question), generate=not args.no_generate)
        _print_response(resp)
        return

    print("FATF-RAG interactive mode. Type a question (or 'quit').")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        _print_response(pipe.query(q, generate=not args.no_generate))


if __name__ == "__main__":
    main()
