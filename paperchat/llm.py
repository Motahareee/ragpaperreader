"""Local instruct model: first-run download, loading, prompting, streaming.

The embedding model is bundled in the app, but the (much larger) instruct
model used to actually answer questions is fetched once on first run and
cached under ~/.cache/paperchat/models/ — after that, everything is fully
offline. The download only ever pulls generic model weights, never any of
the user's documents or questions.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from pathlib import Path

import requests
from llama_cpp import Llama

MODEL_URL = (
    "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/"
    "resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
)
MODEL_FILENAME = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
CACHE_DIR = Path.home() / ".cache" / "paperchat" / "models"

# Shared warning: the excerpt text pulled from real papers often contains the
# *source paper's own* inline reference markers (e.g. "long short-term memory
# [13]"). Without an explicit warning, the model tends to reproduce those
# numbers as if they were our excerpt citations, producing citation badges
# that point at nothing. Excerpt numbering must only ever be the [1]..[N]
# scheme assigned to the excerpts below.
_CITATION_WARNING = (
    "Some excerpts contain the ORIGINAL paper's own inline reference numbers "
    "(e.g. '[13]') as part of their text -- these are not your citation "
    "scheme and must never be reproduced. The only valid citation numbers "
    "are the excerpt numbers [1] through [N] exactly as labeled in the "
    "'Excerpts'/'Source excerpts' list below."
)

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about a specific set of "
    "papers. Only use the information in the provided excerpts to answer — if "
    "the excerpts don't contain the answer, say so plainly instead of guessing. "
    "Cite the excerpt(s) you used with bracketed numbers like [1] or [2] "
    "immediately after the relevant claim. " + _CITATION_WARNING
)

RELATED_WORK_SYSTEM_PROMPT = (
    "You are a research assistant helping draft the 'Related Work' section of a "
    "paper. You are given a topic and excerpts from several source papers. Write "
    "a cohesive, well-organized related-work section (roughly 500-700 words, "
    "about one page) that synthesizes these papers in relation to the topic — "
    "group related approaches together, and note agreements, differences, or "
    "gaps between them where relevant. Write flowing academic prose, not a "
    "list. Only use information from the excerpts provided. Cite every specific "
    "claim with the bracketed excerpt number(s) it came from, like [1] or "
    "[2, 5], immediately after the claim. " + _CITATION_WARNING
)

ProgressCallback = Callable[[int, int], None]


def model_path() -> Path:
    return CACHE_DIR / MODEL_FILENAME


def is_model_downloaded() -> bool:
    return model_path().exists()


def download_model(progress_cb: ProgressCallback | None = None) -> Path:
    dest = model_path()
    if dest.exists():
        return dest
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(".part")
    with requests.get(MODEL_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(tmp_dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if progress_cb:
                    progress_cb(written, total)
    tmp_dest.rename(dest)
    return dest


# Matches the source paper's own inline reference markers, e.g. "[13]",
# "[13, 27]", "[13-15]" -- numeric only, so it won't touch things like
# "[Figure 2]". Stripped before the excerpt ever reaches the model: prompt
# instructions alone weren't reliable enough at stopping a small local model
# from reproducing these as if they were our excerpt-citation numbers.
_SOURCE_CITATION_RE = re.compile(r"\[\d+(?:\s*[-–,]\s*\d+)*\]")


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        name = Path(c["doc_path"]).name
        clean_text = _SOURCE_CITATION_RE.sub("", c["text"])
        clean_text = re.sub(r"[ \t]{2,}", " ", clean_text)
        parts.append(f"[{i}] ({name}, page {c['page'] + 1}):\n{clean_text}")
    return "\n\n".join(parts)


def clean_out_of_range_citations(text: str, max_index: int) -> str:
    """Defensive second layer: drop any citation number outside [1, max_index]
    that the model still produced despite the cleaned excerpts + instructions."""

    def _clean(match: re.Match) -> str:
        nums = [n.strip() for n in match.group(1).split(",")]
        kept = [n for n in nums if n.isdigit() and 1 <= int(n) <= max_index]
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _clean, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)  # dangling space before punctuation
    return cleaned.strip()


class AnswerEngine:
    def __init__(self, path: Path | None = None, n_ctx: int = 16384):
        # 16k context leaves headroom for the related-work generator, which
        # packs in excerpts from several papers at once (Q&A only ever needs
        # a handful of chunks, but this keeps one shared context size simple).
        self._llm = Llama(
            model_path=str(path or model_path()),
            n_ctx=n_ctx,
            verbose=False,
        )

    def ask(self, question: str, chunks: list[dict]) -> Iterator[str]:
        context = _format_context(chunks)
        user_msg = f"Excerpts:\n\n{context}\n\nQuestion: {question}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        stream = self._llm.create_chat_completion(messages=messages, stream=True)
        for event in stream:
            delta = event["choices"][0]["delta"]
            token = delta.get("content")
            if token:
                yield token

    def write_related_work(self, topic: str, chunks: list[dict]) -> Iterator[str]:
        context = _format_context(chunks)
        user_msg = f"Topic: {topic}\n\nSource excerpts:\n\n{context}\n\nWrite the related-work section now."
        messages = [
            {"role": "system", "content": RELATED_WORK_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        stream = self._llm.create_chat_completion(messages=messages, stream=True, max_tokens=1200)
        for event in stream:
            delta = event["choices"][0]["delta"]
            token = delta.get("content")
            if token:
                yield token
