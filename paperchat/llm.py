"""Local instruct model: first-run download, loading, prompting, streaming.

The embedding model is bundled in the app, but the (much larger) instruct
model used to actually answer questions is fetched once on first run and
cached under ~/.cache/paperchat/models/ — after that, everything is fully
offline. The download only ever pulls generic model weights, never any of
the user's documents or questions.
"""
from __future__ import annotations

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

SYSTEM_PROMPT = (
    "You are a research assistant answering questions about a specific set of "
    "papers. Only use the information in the provided excerpts to answer — if "
    "the excerpts don't contain the answer, say so plainly instead of guessing. "
    "Cite the excerpt(s) you used with bracketed numbers like [1] or [2] "
    "immediately after the relevant claim."
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


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        name = Path(c["doc_path"]).name
        parts.append(f"[{i}] ({name}, page {c['page'] + 1}):\n{c['text']}")
    return "\n\n".join(parts)


class AnswerEngine:
    def __init__(self, path: Path | None = None, n_ctx: int = 8192):
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
