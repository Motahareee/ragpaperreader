"""Lightweight local vector store.

No external vector database: embeddings are kept as a plain numpy array
plus a JSONL metadata sidecar inside a `.ragindex/` folder next to the
papers. Brute-force cosine similarity is fast enough for a folder of
papers (thousands of chunks fit comfortably in memory).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .indexing import Chunk, discover_pdfs, file_hash, parse_pdf

INDEX_DIRNAME = ".ragindex"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.jsonl"
MANIFEST_FILE = "manifest.json"


class Store:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.index_dir = self.folder / INDEX_DIRNAME
        self.index_dir.mkdir(exist_ok=True)
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.metadata: list[dict] = []
        self.manifest: dict[str, str] = {}
        self._load()

    def _load(self):
        emb_path = self.index_dir / EMBEDDINGS_FILE
        meta_path = self.index_dir / METADATA_FILE
        manifest_path = self.index_dir / MANIFEST_FILE

        if emb_path.exists():
            self.embeddings = np.load(emb_path)
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = [json.loads(line) for line in f if line.strip()]
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())

    def _save(self):
        np.save(self.index_dir / EMBEDDINGS_FILE, self.embeddings)
        with open(self.index_dir / METADATA_FILE, "w") as f:
            for m in self.metadata:
                f.write(json.dumps(m) + "\n")
        (self.index_dir / MANIFEST_FILE).write_text(json.dumps(self.manifest))

    def needs_reindex(self, pdf_paths: list[Path]) -> list[Path]:
        stale = []
        for p in pdf_paths:
            h = file_hash(p)
            if self.manifest.get(str(p.resolve())) != h:
                stale.append(p)
        return stale

    def remove_document(self, doc_path: str):
        keep_idx = [i for i, m in enumerate(self.metadata) if m["doc_path"] != doc_path]
        if len(keep_idx) == len(self.metadata):
            return
        self.metadata = [self.metadata[i] for i in keep_idx]
        if self.embeddings.shape[0]:
            self.embeddings = self.embeddings[keep_idx]

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray):
        if not chunks:
            return
        self.metadata.extend(asdict(c) for c in chunks)
        if self.embeddings.shape[0] == 0:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])

    def mark_indexed(self, pdf_path: Path):
        self.manifest[str(pdf_path.resolve())] = file_hash(pdf_path)

    def prune_deleted(self, current_pdf_paths: list[Path]):
        current = {str(p.resolve()) for p in current_pdf_paths}
        for known_path in list(self.manifest):
            if known_path not in current:
                self.remove_document(known_path)
                del self.manifest[known_path]

    def save(self):
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.embeddings.shape[0] == 0:
            return []
        scores = self.embeddings @ query_vector
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for i in top_idx:
            item = dict(self.metadata[i])
            item["score"] = float(scores[i])
            results.append(item)
        return results

    @property
    def num_chunks(self) -> int:
        return len(self.metadata)


def sync_folder(folder: Path, embedder) -> Store:
    """Index new/changed PDFs in `folder` and drop removed ones. Returns the Store."""
    store = Store(folder)
    pdf_paths = discover_pdfs(folder)
    store.prune_deleted(pdf_paths)

    stale = store.needs_reindex(pdf_paths)
    for pdf_path in stale:
        store.remove_document(str(pdf_path.resolve()))
        chunks = parse_pdf(pdf_path)
        if chunks:
            vectors = embedder.embed([c.text for c in chunks])
            store.add_chunks(chunks, vectors)
        store.mark_indexed(pdf_path)

    if stale:
        store.save()
    return store
