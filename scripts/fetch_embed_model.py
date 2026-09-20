"""One-time helper to fetch the small embedding GGUF model bundled with the app.

This is run by the developer before packaging (the resulting file is
shipped inside PaperChat.app), not by end users at runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

# Small (~45MB), CPU-friendly sentence embedding model, GGUF build with
# pooling baked in so llama.cpp returns ready-to-use sentence vectors.
MODEL_URL = (
    "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/"
    "resolve/main/all-MiniLM-L6-v2-Q8_0.gguf"
)
DEST = Path(__file__).resolve().parent.parent / "models" / "embed-model.gguf"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST}")
        return
    print(f"downloading {MODEL_URL} -> {DEST}")
    with requests.get(MODEL_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(DEST, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = written / total * 100
                    print(f"\r{pct:5.1f}%", end="", file=sys.stderr)
    print()
    print("done")


if __name__ == "__main__":
    main()
