"""Deterministic, dependency-free embedder for OFFLINE testing/CI.

It hashes character n-grams into a fixed-dim L2-normalised vector. It is NOT a
good semantic model — it is only here so the pipeline, FAISS store, hybrid
fusion, and evaluation harness can be exercised without downloading model
weights (e.g. in a sandbox with no Hugging Face access). Production runs use
src.embeddings.Embedder (sentence-transformers).
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np


class HashingEmbedder:
    def __init__(self, dim: int = 384, model_name: str = "hashing-test-embedder"):
        self.dim = dim
        self.model_name = model_name

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        toks = re.findall(r"[a-z0-9]+", text.lower())
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self._vec(text)
