"""Local open-source generators (GPT-2 / Flan-T5) via Hugging Face transformers.

Ported from the group's experimental notebook. This backend runs entirely on
your machine with no API key and no Ollama — it only needs `transformers` and
`torch` (already pulled in by sentence-transformers). It is the backend used by
the 6-scenario comparison in `eval/scenarios.py`.

Two model families are supported, exactly as in the experiments:
  * "gpt2"    — causal LM; the output contains the prompt, so we strip it and
                keep prompt + generation within GPT-2's 1024-token window.
  * "flan-t5" — seq2seq LM; the output IS the answer (no prompt echoed back).

Generation is seeded for reproducibility.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

try:
    import torch
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        T5ForConditionalGeneration,
        T5Tokenizer,
        set_seed,
    )
    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    _HAS_TRANSFORMERS = False

_DEVICE = None


def _device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


_MODEL_IDS = {"gpt2": "gpt2", "flan-t5": "google/flan-t5-base"}


@lru_cache(maxsize=2)
def load_generator(model_type: str):
    """Load and cache (tokenizer, model) for 'gpt2' or 'flan-t5'."""
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers/torch are required for the HF generator backend.")
    if model_type == "gpt2":
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(_device())
        tok.pad_token = tok.eos_token
    elif model_type == "flan-t5":
        tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(_device())
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'gpt2' or 'flan-t5'.")
    return tok, model


def generate_answer(
    prompt: str, model_type: str, max_new_tokens: int = 200, seed: int = 42
) -> str:
    """Generate an answer string from an augmented prompt (see llm.create_augmented_prompt)."""
    tokenizer, model = load_generator(model_type)
    set_seed(seed)

    if model_type == "gpt2":
        max_input_tokens = 1024 - max_new_tokens
        input_ids = tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
        ).to(_device())
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        full = tokenizer.decode(output[0], skip_special_tokens=True)
        return full[len(prompt):].strip()

    # flan-t5 (seq2seq)
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(_device())
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def available() -> bool:
    return _HAS_TRANSFORMERS
