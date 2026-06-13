"""Local embedding model wrapper.

We use sentence-transformers (all-MiniLM-L6-v2 by default) because:
  * it runs locally with no API key -> reproducible and free for the evaluator;
  * 384-dim vectors keep the FAISS index tiny and fast on CPU;
  * it is a well-understood baseline, so design trade-offs are easy to defend.

Vectors are L2-normalised so that inner-product search in FAISS is equivalent to
cosine similarity.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np


@lru_cache(maxsize=2)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = _load_model(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # -> cosine == inner product
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
