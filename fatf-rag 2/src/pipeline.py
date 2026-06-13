"""End-to-end RAG orchestration: load index -> retrieve -> (optionally) generate.

This is the single object the CLI, the Streamlit app, and the notebook all use,
so behaviour is identical across every entry-point.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import Config, DEFAULT
from .embeddings import Embedder
from .ingest import build_chunks
from .llm import LLM
from .retriever import Retriever, RetrievedChunk
from .vectorstore import VectorStore


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[RetrievedChunk]

    def format_sources(self) -> str:
        lines = []
        for i, s in enumerate(self.sources, 1):
            lines.append(f"{i}. {s.citation}  (score={s.score:.3f})\n   {s.chunk.text[:220]}…")
        return "\n".join(lines)


def _context_block(sources: List[RetrievedChunk]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(f"[Passage {i}] {s.citation}\n{s.chunk.text}")
    return "\n\n".join(parts)


class RAGPipeline:
    def __init__(
        self,
        cfg: Config = DEFAULT,
        store: Optional[VectorStore] = None,
        embedder=None,  # inject a custom embedder (e.g. for offline tests)
    ):
        self.cfg = cfg
        self.embedder = embedder if embedder is not None else Embedder(cfg.embed_model)
        if store is None:
            store = VectorStore.load(cfg.index_path, cfg.chunks_path)
        self.store = store
        self.retriever = Retriever(
            store, self.embedder, mode=cfg.retriever, hybrid_alpha=cfg.hybrid_alpha
        )
        self.llm = LLM(cfg)

    # --- factory helpers ---
    @classmethod
    def from_storage(cls, cfg: Config = DEFAULT) -> "RAGPipeline":
        return cls(cfg)

    @classmethod
    def build_from_pdf(cls, cfg: Config = DEFAULT, save: bool = True, embedder=None) -> "RAGPipeline":
        chunks = build_chunks(
            cfg.pdf_path, cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_chars
        )
        embedder = embedder if embedder is not None else Embedder(cfg.embed_model)
        embs = embedder.encode([c.text for c in chunks])
        store = VectorStore(chunks, embs)
        if save:
            store.save(cfg.index_path, cfg.chunks_path)
        return cls(cfg, store=store, embedder=embedder)

    # --- query ---
    def query(self, question: str, top_k: Optional[int] = None, generate: bool = True) -> RAGResponse:
        k = top_k or self.cfg.top_k
        sources = self.retriever.retrieve(question, top_k=k)
        if generate:
            answer = self.llm.generate(question, _context_block(sources))
        else:
            answer = _context_block(sources)
        return RAGResponse(question=question, answer=answer, sources=sources)
