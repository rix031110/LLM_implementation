"""Central configuration. All paths are repo-relative so the project is portable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"
EVAL_DIR = ROOT / "eval"

PDF_PATH = DATA_DIR / "ENG_REC.pdf"


@dataclass
class Config:
    # --- Ingestion / chunking ---
    pdf_path: Path = PDF_PATH
    chunk_size: int = 900           # characters per chunk (~180-220 tokens)
    chunk_overlap: int = 150        # overlap keeps cross-boundary context
    min_chunk_chars: int = 80       # drop boilerplate fragments smaller than this

    # --- Embeddings ---
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # --- Retrieval ---
    top_k: int = 5                  # chunks returned to the LLM
    hybrid_alpha: float = 0.5       # weight on dense vs BM25 in hybrid fusion (1=dense only)
    retriever: str = "hybrid"       # one of: dense | bm25 | hybrid

    # --- Generation (fully local — no API key) ---
    # backend: "hf" (transformers GPT-2/Flan-T5, works with just pip) or
    #          "ollama" (local Llama via Ollama, higher quality, needs install).
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "hf"))
    hf_model: str = field(default_factory=lambda: os.getenv("HF_MODEL", "flan-t5"))  # flan-t5 | gpt2
    # Ollama settings (used only when llm_backend == "ollama"):
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    temperature: float = 0.0

    # --- Storage ---
    storage_dir: Path = STORAGE_DIR

    @property
    def index_path(self) -> Path:
        return self.storage_dir / "index.faiss"

    @property
    def chunks_path(self) -> Path:
        return self.storage_dir / "chunks.pkl"


DEFAULT = Config()
