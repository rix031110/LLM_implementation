"""Streamlit web interface for FATF-RAG.

Run:  streamlit run app/streamlit_app.py

Lets you pick the retriever AND the generation model (Flan-T5, GPT-2, or
Llama via Ollama) live from the sidebar — no .env editing needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when Streamlit runs from app/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.config import Config
from src.llm import LLM
from src.pipeline import RAGPipeline

st.set_page_config(page_title="FATF-RAG", page_icon="🔎", layout="wide")

# Map a friendly UI label -> (backend, model field overrides).
GEN_CHOICES = {
    "Flan-T5  (local, transformers)": {"llm_backend": "hf", "hf_model": "flan-t5"},
    "GPT-2  (local, transformers)": {"llm_backend": "hf", "hf_model": "gpt2"},
    "Llama 3.2  (local, Ollama)": {"llm_backend": "ollama", "llm_model": "llama3.2"},
}


@st.cache_resource(show_spinner="Loading index and embedding model…")
def load_pipeline(retriever: str, alpha: float):
    """Cached: holds the index, embedder and retriever (the heavy parts)."""
    cfg = Config()
    cfg.retriever = retriever
    cfg.hybrid_alpha = alpha
    try:
        return RAGPipeline.from_storage(cfg), None
    except FileNotFoundError:
        return None, "No index found. Run `python -m scripts.build_index` first."


def make_llm(choice: str) -> LLM:
    """Build a generation backend from the sidebar choice (cheap; models load lazily)."""
    cfg = Config()
    for k, v in GEN_CHOICES[choice].items():
        setattr(cfg, k, v)
    return LLM(cfg)


st.title("🔎 FATF-RAG — Q&A on the FATF Recommendations")
st.caption(
    "Retrieval-augmented QA over the FATF 40 Recommendations (AML/CFT standard, Feb 2012). "
    "Answers are grounded in retrieved passages with Recommendation + page citations."
)

with st.sidebar:
    st.header("Retrieval")
    retriever = st.radio("Retriever", ["hybrid", "dense", "bm25"], index=0)
    alpha = st.slider("Hybrid α (dense weight)", 0.0, 1.0, 0.5, 0.05,
                      help="1.0 = pure dense, 0.0 = pure BM25. Used in hybrid mode only.")
    top_k = st.slider("Passages (top-k)", 1, 10, 5)

    st.divider()
    st.header("Generation")
    generate = st.toggle("Generate an answer", value=True,
                         help="Off = show retrieved passages only.")
    gen_choice = st.radio("Model", list(GEN_CHOICES.keys()), index=0,
                          help="Flan-T5 / GPT-2 run locally via transformers (just pip). "
                               "Llama needs Ollama running (ollama pull llama3.2).")

pipe, err = load_pipeline(retriever, alpha)
if err:
    st.error(err)
    st.stop()

# Swap in the chosen generation backend (cheap — the model itself loads on first use).
if generate:
    pipe.llm = make_llm(gen_choice)
    if pipe.llm.available:
        st.sidebar.success(f"Active model: {gen_choice.split('  ')[0]}")
    else:
        hint = ("run `pip install -r requirements.txt`" if pipe.llm.backend == "hf"
                else "install Ollama and run `ollama pull llama3.2`")
        st.warning(f"'{gen_choice.split('  ')[0]}' isn't available yet — {hint}. "
                   "Showing retrieved passages only for now.")
        generate = False

examples = [
    "What is a beneficial owner under the FATF Recommendations?",
    "When must enhanced due diligence be applied?",
    "What are the wire transfer requirements in Recommendation 16?",
    "What is the risk-based approach?",
]
ex = st.selectbox("Try an example question", [""] + examples)
question = st.text_input("Your question", value=ex)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving…" if not generate else f"Retrieving + generating with {gen_choice.split('  ')[0]}…"):
        resp = pipe.query(question, top_k=top_k, generate=generate)
    if generate:
        st.subheader("Answer")
        st.write(resp.answer)
        st.caption(f"Generated with {gen_choice.split('  ')[0]}")
    st.subheader("Sources")
    for i, s in enumerate(resp.sources, 1):
        with st.expander(f"{i}. {s.citation}  ·  score={s.score:.3f}"):
            st.write(s.chunk.text)
