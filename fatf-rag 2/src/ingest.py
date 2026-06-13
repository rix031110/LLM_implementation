"""PDF ingestion + structure-aware chunking for the FATF Recommendations.

Design notes
------------
The FATF Recommendations is a highly structured legal document: 40 numbered
Recommendations, a block of Interpretive Notes, and a Glossary. Two facts drive
the chunking strategy:

1. Page numbers matter for citation. A user (or evaluator) needs to verify an
   answer against the source, so every chunk carries its page span.
2. The document has natural section anchors ("Recommendation 10", "INTERPRETIVE
   NOTE TO RECOMMENDATION 10", "A. ...", glossary terms). We tag each chunk with
   the most recent detected heading so retrieval and answers can name the
   governing Recommendation.

We deliberately keep chunking simple and transparent (recursive character split
with overlap) rather than a heavyweight semantic splitter: it is deterministic,
fast, and easy to defend in the write-up. The chunking parameters are tuned in
the evaluation (see eval/evaluate.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pdfplumber

# Headings that anchor a chunk to a part of the document.
_HEADING_PATTERNS = [
    re.compile(r"^\s*INTERPRETIVE NOTE TO RECOMMENDATION\s+(\d+)", re.I),
    re.compile(r"^\s*RECOMMENDATION\s+(\d+)\b", re.I),
    re.compile(r"^\s*(\d+)\.\s+[A-Z][A-Za-z].{0,80}$"),  # "10. Customer due diligence"
    re.compile(r"^\s*(GLOSSARY|GENERAL GLOSSARY)\b", re.I),
]


@dataclass
class Chunk:
    id: int
    text: str
    page_start: int
    page_end: int
    section: str  # best-effort heading label, e.g. "Recommendation 10"

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    """Normalise whitespace and strip running headers/footers heuristically."""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        # Drop pure page-number lines and the recurring copyright footer.
        if re.fullmatch(r"\d{1,3}", s):
            continue
        if "FATF/OECD" in s or "FATF Recommendations" == s:
            continue
        lines.append(s)
    return "\n".join(lines)


def _detect_section(text: str, current: str) -> str:
    """Return an updated section label if a heading appears in `text`."""
    for line in text.splitlines()[:6]:
        for pat in _HEADING_PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            g = m.group(1)
            if g and g.isdigit():
                label = f"Recommendation {g}"
                if "INTERPRETIVE" in line.upper():
                    label = f"Interpretive Note to Recommendation {g}"
                return label
            if g:
                return g.title()
    return current


def load_pages(pdf_path: str | Path) -> List[tuple[int, str]]:
    """Return [(page_number, cleaned_text), ...] (1-indexed pages)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Corpus PDF not found at {pdf_path}. Place ENG_REC.pdf in data/."
        )
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            pages.append((i, _clean(raw)))
    return pages


def _split_with_overlap(text: str, size: int, overlap: int) -> List[str]:
    """Recursive-ish character splitter that prefers paragraph/sentence breaks."""
    if len(text) <= size:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Try to break on a paragraph, then sentence, then space.
            window = text[start:end]
            for sep in ("\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > size * 0.5:  # only honour a break in the latter half
                    end = start + cut + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def _build_page_chunks(pdf_path: str | Path) -> List[Chunk]:
    """One page = one chunk (the group's 'page' chunking scenario)."""
    chunks: List[Chunk] = []
    section = "Front matter"
    for page_no, text in load_pages(pdf_path):
        if not text:
            continue
        section = _detect_section(text, section)
        chunks.append(Chunk(id=len(chunks), text=text, page_start=page_no,
                            page_end=page_no, section=section))
    return chunks


def _build_paragraph_chunks(
    pdf_path: str | Path, words_per_chunk: int = 300, overlap_words: int = 0
) -> List[Chunk]:
    """Word-based sliding window over each page (the group's 'paragraph' scenario).

    Mirrors the notebook's build_knowledge_base(strategy='paragraph'): each page
    is flattened to words, then a window of `words_per_chunk` slides forward by
    `words_per_chunk - overlap_words`.
    """
    chunks: List[Chunk] = []
    section = "Front matter"
    step = max(1, words_per_chunk - overlap_words)
    for page_no, text in load_pages(pdf_path):
        if not text:
            continue
        words = []
        for para in (p.strip() for p in text.split("\n")):
            if para:
                words.extend(para.split())
        if not words:
            continue
        for start in range(0, len(words), step):
            sub = words[start:start + words_per_chunk]
            if not sub:
                break
            piece = " ".join(sub)
            section = _detect_section(piece, section)
            chunks.append(Chunk(id=len(chunks), text=piece, page_start=page_no,
                                page_end=page_no, section=section))
            if start + words_per_chunk >= len(words):
                break
    return chunks


def build_chunks(
    pdf_path: str | Path,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 80,
    strategy: str = "recursive",
) -> List[Chunk]:
    """Full ingestion with a selectable chunking strategy.

    strategy:
      * "recursive"  (default) — character-based recursive split with overlap;
                      `chunk_size`/`chunk_overlap` are in CHARACTERS. Used by the
                      production pipeline.
      * "page"       — one page = one chunk (the group's 'page' scenario).
      * "paragraph"  — word-based sliding window over each page (the group's
                      'paragraph' scenario); here `chunk_size`/`chunk_overlap`
                      are interpreted in WORDS.

    All strategies return Chunk objects with page metadata so retrieval,
    citation, and evaluation work identically downstream.
    """
    if strategy == "page":
        return _build_page_chunks(pdf_path)
    if strategy == "paragraph":
        return _build_paragraph_chunks(pdf_path, words_per_chunk=chunk_size, overlap_words=chunk_overlap)
    if strategy != "recursive":
        raise ValueError(f"Unknown strategy: '{strategy}'. Use recursive|page|paragraph.")

    pages = load_pages(pdf_path)
    # Concatenate page text but remember page boundaries via character offsets.
    full_text = ""
    page_offsets: list[tuple[int, int]] = []  # (char_offset, page_number)
    for page_no, text in pages:
        if not text:
            continue
        page_offsets.append((len(full_text), page_no))
        full_text += text + "\n\n"

    def page_at(offset: int) -> int:
        page = page_offsets[0][1] if page_offsets else 1
        for off, pno in page_offsets:
            if off <= offset:
                page = pno
            else:
                break
        return page

    raw_chunks = _split_with_overlap(full_text, chunk_size, chunk_overlap)

    chunks: List[Chunk] = []
    section = "Front matter"
    cursor = 0
    for piece in raw_chunks:
        # Locate this piece in full_text to recover its page span.
        idx = full_text.find(piece[:40], cursor) if piece else -1
        if idx == -1:
            idx = cursor
        start_page = page_at(idx)
        end_page = page_at(idx + len(piece))
        cursor = idx + max(len(piece) - chunk_overlap, 1)

        section = _detect_section(piece, section)
        if len(piece) < min_chunk_chars:
            continue
        chunks.append(
            Chunk(
                id=len(chunks),
                text=piece,
                page_start=start_page,
                page_end=end_page,
                section=section,
            )
        )
    return chunks


if __name__ == "__main__":  # quick manual check
    from .config import DEFAULT

    cs = build_chunks(DEFAULT.pdf_path, DEFAULT.chunk_size, DEFAULT.chunk_overlap)
    print(f"Built {len(cs)} chunks")
    for c in cs[:3]:
        print(f"[{c.id}] p{c.page_start}-{c.page_end} | {c.section}\n{c.text[:200]}\n---")
