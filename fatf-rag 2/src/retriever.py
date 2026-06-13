"""Retrieval layer: dense, BM25 (lexical), and a hybrid of the two.

Why hybrid?
-----------
Legal text mixes two query styles. Some questions are conceptual ("what is a
risk-based approach?") where dense semantic search wins. Others hinge on exact
terms or numbers ("Recommendation 16", "beneficial owner", "USD/EUR 15,000")
where lexical BM25 is more reliable. Combining them with score normalisation
gives robustness across both, which we confirm quantitatively in the evaluation.

Fusion: we min-max normalise each retriever's scores to [0,1] over the candidate
pool, then combine as  alpha*dense + (1-alpha)*bm25.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .ingest import Chunk
from .vectorstore import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> str:
        c = self.chunk
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
        return f"[{c.section}, {pages}]"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        mode: str = "hybrid",
        hybrid_alpha: float = 0.5,
    ):
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.alpha = hybrid_alpha
        self.chunks = store.chunks
        # Pre-build the BM25 index over chunk tokens.
        self._corpus_tokens = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def _dense_scores(self, query: str) -> np.ndarray:
        qv = self.embedder.encode_one(query).reshape(1, -1)
        # full-corpus cosine via the stored matrix (small corpus -> exact is fine)
        return (self.store.embeddings @ qv.T).ravel()

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(_tokenize(query)), dtype="float32")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if self.mode == "dense":
            scores = self._dense_scores(query)
        elif self.mode == "bm25":
            scores = self._bm25_scores(query)
        elif self.mode == "hybrid":
            d = _minmax(self._dense_scores(query))
            b = _minmax(self._bm25_scores(query))
            scores = self.alpha * d + (1 - self.alpha) * b
        else:
            raise ValueError(f"Unknown retriever mode: {self.mode}")

        order = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in order]
