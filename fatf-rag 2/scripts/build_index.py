"""Build (or rebuild) the vector index from the FATF PDF.

Usage:
    python -m scripts.build_index            # default chunking
    python -m scripts.build_index --chunk-size 1200 --overlap 200
"""
from __future__ import annotations

import argparse
import time

from src.config import Config
from src.pipeline import RAGPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the FATF-RAG vector index.")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--embed-model", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.chunk_size:
        cfg.chunk_size = args.chunk_size
    if args.overlap is not None:
        cfg.chunk_overlap = args.overlap
    if args.embed_model:
        cfg.embed_model = args.embed_model

    print(f"Building index from {cfg.pdf_path}")
    print(f"  chunk_size={cfg.chunk_size} overlap={cfg.chunk_overlap} model={cfg.embed_model}")
    t = time.time()
    pipe = RAGPipeline.build_from_pdf(cfg, save=True)
    n = len(pipe.store.chunks)
    print(f"Indexed {n} chunks in {time.time()-t:.1f}s -> {cfg.index_path}")


if __name__ == "__main__":
    main()
