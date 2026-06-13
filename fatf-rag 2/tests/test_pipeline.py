"""Offline smoke + behaviour tests. Run:  python -m tests.test_pipeline

Uses a hashing embedder (no model download) so it works anywhere. Verifies:
  * ingestion produces non-trivial, page-tagged chunks from the real PDF,
  * FAISS dense search, BM25, and hybrid retrieval all return results,
  * the RAGPipeline runs end-to-end in retrieval-only mode,
  * the evaluation harness computes sane metrics,
  * BM25 (a real production retriever) actually finds relevant pages,
    which is a genuine quality signal independent of the embedding model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.ingest import build_chunks
from src.retriever import Retriever
from src.vectorstore import VectorStore
from src.pipeline import RAGPipeline
from eval.evaluate import evaluate_retriever, load_testset
from tests.offline_embedder import HashingEmbedder


def main() -> int:
    cfg = Config()
    emb = HashingEmbedder(dim=384)
    failures = []

    # 1. Ingestion
    chunks = build_chunks(cfg.pdf_path, cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_chars)
    assert len(chunks) > 50, "too few chunks"
    assert all(c.page_start >= 1 for c in chunks), "bad page metadata"
    sections = {c.section for c in chunks}
    print(f"[1] ingestion: {len(chunks)} chunks, {len(sections)} distinct sections  OK")

    # 2. Vector store + dense search
    embs = emb.encode([c.text for c in chunks])
    store = VectorStore(chunks, embs)
    assert store.ntotal == len(chunks)
    print(f"[2] vector index: {store.ntotal} vectors, dim={store.dim}  OK")

    # 3. All three retrievers return top-k
    for mode in ["dense", "bm25", "hybrid"]:
        r = Retriever(store, emb, mode=mode).retrieve("beneficial owner definition", top_k=5)
        assert len(r) == 5, f"{mode} returned {len(r)}"
        print(f"[3] retriever[{mode}] top page span: {r[0].chunk.page_start}-{r[0].chunk.page_end}  OK")

    # 4. Pipeline end-to-end (retrieval-only, no LLM key needed)
    pipe = RAGPipeline(cfg, store=store, embedder=emb)
    resp = pipe.query("What is enhanced due diligence?", generate=False)
    assert len(resp.sources) == cfg.top_k
    assert resp.sources[0].citation.startswith("[")
    print(f"[4] pipeline end-to-end: {len(resp.sources)} sources, sample cite {resp.sources[0].citation}  OK")

    # 5. Evaluation harness + BM25 quality signal
    ts = load_testset()
    bm25 = Retriever(store, emb, mode="bm25")
    res = evaluate_retriever(bm25, ts["questions"], k=5)
    print(f"[5] BM25 eval: Recall@5={res['recall@k']}  MRR={res['mrr']}  P@5={res['precision@k']}")
    if res["recall@k"] < 0.5:
        failures.append(f"BM25 Recall@5 unexpectedly low: {res['recall@k']}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL OFFLINE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
