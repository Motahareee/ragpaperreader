"""Local embedding model wrapper.

Uses a small GGUF embedding model via llama-cpp-python so the app never
needs a torch/sentence-transformers dependency (which is painful to bundle
with PyInstaller on macOS). The embedding model is small enough (tens of
MB) to ship inside the packaged app itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from llama_cpp import Llama

from .paths import resource_root

DEFAULT_EMBED_MODEL_PATH = resource_root() / "models" / "embed-model.gguf"


class Embedder:
    def __init__(self, model_path: Path | None = None, n_ctx: int = 512):
        model_path = model_path or DEFAULT_EMBED_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found at {model_path}. "
                "Run scripts/fetch_embed_model.py or place a GGUF embedding model there."
            )
        self._llm = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=n_ctx,
            verbose=False,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) L2-normalized float32 embedding matrix."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        result = self._llm.create_embedding(texts)
        vectors = [np.array(item["embedding"], dtype=np.float32) for item in result["data"]]
        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
