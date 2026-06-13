"""Vector store + on-disk persistence of chunks and embeddings.

Backend: pure NumPy exact cosine search by default (the corpus is only ~400
chunks, so brute-force inner product is instant and adds no dependency). If
FAISS is installed it is used transparently as an accelerator, but it is fully
optional — this keeps installation painless across Python versions / platforms
where a faiss-cpu wheel may be unavailable.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

from .ingest import Chunk

try:  # optional acceleration
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray):
        assert len(chunks) == embeddings.shape[0], "chunks/embeddings length mismatch"
        self.chunks = chunks
        self.embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        self.dim = self.embeddings.shape[1]
        self._build_index()

    def _build_index(self) -> None:
        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine via normalised vectors
            self.index.add(self.embeddings)
        else:
            self.index = None  # NumPy fallback uses self.embeddings directly

    @property
    def ntotal(self) -> int:
        """Number of indexed vectors (works with or without faiss)."""
        return self.index.ntotal if _HAS_FAISS and self.index is not None else len(self.chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        if _HAS_FAISS and self.index is not None:
            scores, idxs = self.index.search(q, top_k)
            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
        # NumPy exact cosine (embeddings are L2-normalised -> dot == cosine)
        sims = (self.embeddings @ q.T).ravel()
        top = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in top]

    # --- persistence ---
    def save(self, index_path: Path, chunks_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, str(index_path))
        with open(chunks_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, index_path: Path, chunks_path: Path) -> "VectorStore":
        with open(chunks_path, "rb") as f:
            blob = pickle.load(f)
        # Rebuild the store (and its index) from the persisted embeddings. This
        # is robust whether or not faiss is present, so a faiss-built index can
        # be loaded on a faiss-less machine and vice versa.
        return cls(blob["chunks"], blob["embeddings"])
