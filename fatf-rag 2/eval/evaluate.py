"""Retrieval evaluation for FATF-RAG.

Metrics (page-level relevance: a retrieved chunk is relevant if its page span
intersects the question's gold_pages):

  * Recall@k  — fraction of questions with >=1 relevant chunk in the top-k.
                (For single-target IR this equals Hit Rate / Success@k.)
  * MRR       — mean reciprocal rank of the first relevant chunk.
  * Precision@k — average fraction of the top-k chunks that are relevant.

It also runs an ABLATION across retrievers (dense / bm25 / hybrid) and across
chunk sizes, so the report can justify the chosen configuration with numbers
rather than assertion.

Usage:
    python -m eval.evaluate                 # evaluate the saved index (current config)
    python -m eval.evaluate --ablation      # rebuild + compare configs (slower)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from src.config import Config
from src.embeddings import Embedder
from src.ingest import build_chunks
from src.retriever import Retriever
from src.vectorstore import VectorStore

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def load_testset() -> dict:
    with open(EVAL_DIR / "testset.json") as f:
        return json.load(f)


def _pages(chunk) -> set:
    return set(range(chunk.page_start, chunk.page_end + 1))


def _is_relevant(chunk, gold_pages: List[int]) -> bool:
    return bool(_pages(chunk) & set(gold_pages))


def evaluate_retriever(retriever: Retriever, questions: List[dict], k: int = 5) -> dict:
    recall_hits = 0
    rr_sum = 0.0
    prec_sum = 0.0
    per_q = []
    for q in questions:
        results = retriever.retrieve(q["question"], top_k=k)
        rels = [_is_relevant(r.chunk, q["gold_pages"]) for r in results]
        hit = any(rels)
        recall_hits += int(hit)
        rr = 0.0
        for rank, is_rel in enumerate(rels, 1):
            if is_rel:
                rr = 1.0 / rank
                break
        rr_sum += rr
        prec = sum(rels) / k
        prec_sum += prec
        per_q.append({
            "id": q["id"], "hit": hit, "rr": round(rr, 3),
            "precision@k": round(prec, 3),
            "top_pages": [f"{r.chunk.page_start}-{r.chunk.page_end}" for r in results],
        })
    n = len(questions)
    return {
        "n": n, "k": k,
        "recall@k": round(recall_hits / n, 3),
        "mrr": round(rr_sum / n, 3),
        "precision@k": round(prec_sum / n, 3),
        "per_question": per_q,
    }


def build_store(cfg: Config, embedder: Embedder) -> VectorStore:
    chunks = build_chunks(cfg.pdf_path, cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_chars)
    embs = embedder.encode([c.text for c in chunks])
    return VectorStore(chunks, embs)


def run_default(cfg: Config, k: int = 5) -> dict:
    ts = load_testset()
    embedder = Embedder(cfg.embed_model)
    try:
        store = VectorStore.load(cfg.index_path, cfg.chunks_path)
    except FileNotFoundError:
        store = build_store(cfg, embedder)
    retr = Retriever(store, embedder, mode=cfg.retriever, hybrid_alpha=cfg.hybrid_alpha)
    res = evaluate_retriever(retr, ts["questions"], k=k)
    print(f"\n[{cfg.retriever}] n={res['n']} k={k}  "
          f"Recall@{k}={res['recall@k']}  MRR={res['mrr']}  P@{k}={res['precision@k']}")
    return res


def run_ablation(cfg: Config, k: int = 5) -> dict:
    ts = load_testset()
    embedder = Embedder(cfg.embed_model)
    out = {"k": k, "retriever_comparison": {}, "chunk_size_comparison": {}}

    print("\n=== Retriever comparison (chunk_size=%d) ===" % cfg.chunk_size)
    store = build_store(cfg, embedder)
    for mode in ["bm25", "dense", "hybrid"]:
        retr = Retriever(store, embedder, mode=mode, hybrid_alpha=cfg.hybrid_alpha)
        r = evaluate_retriever(retr, ts["questions"], k=k)
        out["retriever_comparison"][mode] = {kk: r[kk] for kk in ["recall@k", "mrr", "precision@k"]}
        print(f"  {mode:7s}  Recall@{k}={r['recall@k']}  MRR={r['mrr']}  P@{k}={r['precision@k']}")

    print("\n=== Chunk-size comparison (hybrid retriever) ===")
    for cs in [600, 900, 1200]:
        c2 = Config()
        c2.chunk_size = cs
        c2.embed_model = cfg.embed_model
        st = build_store(c2, embedder)
        retr = Retriever(st, embedder, mode="hybrid", hybrid_alpha=cfg.hybrid_alpha)
        r = evaluate_retriever(retr, ts["questions"], k=k)
        out["chunk_size_comparison"][cs] = {
            "n_chunks": len(st.chunks),
            **{kk: r[kk] for kk in ["recall@k", "mrr", "precision@k"]},
        }
        print(f"  size={cs:4d} ({len(st.chunks):3d} chunks)  "
              f"Recall@{k}={r['recall@k']}  MRR={r['mrr']}  P@{k}={r['precision@k']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    cfg = Config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t = time.time()
    if args.ablation:
        results = run_ablation(cfg, k=args.k)
        out = RESULTS_DIR / "ablation.json"
    else:
        results = run_default(cfg, k=args.k)
        out = RESULTS_DIR / "default.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}  ({time.time()-t:.1f}s)")


if __name__ == "__main__":
    main()
