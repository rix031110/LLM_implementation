"""Answer-generation layer with two fully-local backends (no API keys).

Backends (set via LLM_BACKEND in .env / Config.llm_backend):
  * "hf"     — open-source GPT-2 or Flan-T5 through Hugging Face transformers.
               Runs with just `pip install` (no extra services). This is the
               group's experimental backend and the default here, so the project
               runs end-to-end out of the box. Model chosen via HF_MODEL
               ("flan-t5" or "gpt2").
  * "ollama" — a local Llama model served by Ollama at localhost:11434. Higher
               answer quality; requires installing Ollama + `ollama pull`.

Both answer strictly from retrieved context. If the selected backend isn't
available, the pipeline degrades gracefully to retrieval-only mode with an
actionable message instead of crashing.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import Config

# Instruction-style prompt used for the transformers backend (works well for
# Flan-T5 and is harmless for GPT-2). Ported from the group's notebook.
AUGMENTED_PROMPT = """Answer the question using only the context below.

Context: {context}
Question: {question}
Answer:"""

# System prompt for the chat-style Ollama backend.
SYSTEM_PROMPT = (
    "You are a compliance assistant answering questions strictly about the FATF "
    "Recommendations (the global AML/CFT standard). Use ONLY the provided context "
    "passages. If the answer is not in the context, say you cannot find it in the "
    "provided sources. Cite the Recommendation number and page for every claim "
    "using the bracketed citations shown with each passage. Be precise and concise."
)


def create_augmented_prompt(question: str, context_block: str, max_context_chars: int = 900) -> str:
    """Build the instruction prompt, truncating context to a character budget."""
    context = context_block[:max_context_chars]
    return AUGMENTED_PROMPT.format(context=context, question=question)


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backend = cfg.llm_backend
        self.host = cfg.ollama_host.rstrip("/")
        self.model = cfg.llm_model
        self.hf_model = cfg.hf_model

    # ---------- availability ----------
    @property
    def available(self) -> bool:
        if self.backend == "hf":
            from . import hf_generator
            return hf_generator.available()
        return self._ollama_available()

    def _ollama_get(self, path: str, timeout: float = 2.0):
        with urllib.request.urlopen(urllib.request.Request(self.host + path), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _ollama_available(self) -> bool:
        try:
            tags = self._ollama_get("/api/tags")
        except Exception:
            return False
        names = {m.get("name", "").split(":")[0] for m in tags.get("models", [])}
        return self.model.split(":")[0] in names

    def _hint(self) -> str:
        if self.backend == "hf":
            return ("[retrieval-only mode — transformers/torch not installed]\n"
                    "Run: pip install -r requirements.txt\n")
        try:
            self._ollama_get("/api/tags")
            return (f"[retrieval-only mode — model '{self.model}' not found in Ollama]\n"
                    f"Pull it with:   ollama pull {self.model}\n")
        except Exception:
            return (f"[retrieval-only mode — Ollama not reachable at {self.host}]\n"
                    "Install Ollama (https://ollama.com/download) and run:\n"
                    f"    ollama pull {self.model}\n")

    # ---------- generation ----------
    def generate(self, question: str, context_block: str) -> str:
        if not self.available:
            return self._hint() + "\nTop retrieved passages:\n" + context_block
        if self.backend == "hf":
            from . import hf_generator
            prompt = create_augmented_prompt(question, context_block)
            return hf_generator.generate_answer(prompt, self.hf_model)
        return self._ollama_generate(question, context_block)

    def _ollama_generate(self, question: str, context_block: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context passages:\n{context_block}\n\nQuestion: {question}"},
            ],
        }
        req = urllib.request.Request(
            self.host + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read().decode("utf-8"))
            return resp.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as e:  # pragma: no cover
            return f"[generation failed talking to Ollama: {e}]\n\n" + context_block
