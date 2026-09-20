"""Python object exposed to the JS frontend via pywebview's JS-API bridge.

No HTTP server: pywebview lets JS call methods on this object directly
(`pywebview.api.ask_question(...)`), and this object pushes async updates
(indexing/download progress, streamed answer tokens) back into the page
via `window.evaluate_js`.
"""
from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

from . import llm
from .embeddings import Embedder
from .indexing import discover_pdfs
from .store import sync_folder


class Api:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.window = None
        self.embedder: Embedder | None = None
        self.engine: llm.AnswerEngine | None = None
        self.store = None
        self.ready = False

    def set_window(self, window):
        self.window = window

    def _push(self, js_fn: str, *args):
        if not self.window:
            return
        arg_str = ", ".join(json.dumps(a) for a in args)
        self.window.evaluate_js(f"{js_fn}({arg_str})")

    def startup(self):
        threading.Thread(target=self._startup_worker, daemon=True).start()

    def _startup_worker(self):
        try:
            self._push("onStatus", "Loading embedding model...")
            self.embedder = Embedder()

            self._push("onStatus", "Indexing papers in this folder...")
            self.store = sync_folder(self.folder, self.embedder)

            if not llm.is_model_downloaded():
                self._push(
                    "onStatus",
                    "Downloading language model (first run only, ~4.5GB)...",
                )

                def progress(written, total):
                    pct = (written / total * 100) if total else 0
                    self._push("onDownloadProgress", pct)

                llm.download_model(progress)

            self._push("onStatus", "Loading language model...")
            self.engine = llm.AnswerEngine()

            self.ready = True
            self._push("onReady", self.list_papers())
        except Exception as e:  # surface any startup failure to the UI
            self._push("onError", str(e))

    def list_papers(self) -> list[dict]:
        return [{"name": p.name, "path": str(p.resolve())} for p in discover_pdfs(self.folder)]

    def get_pdf_data(self, doc_path: str) -> str:
        """Return a PDF's bytes as base64 so the frontend can hand them to pdf.js
        without needing an HTTP server."""
        return base64.b64encode(Path(doc_path).read_bytes()).decode("ascii")

    def ask_question(self, question: str) -> dict:
        if not self.ready:
            return {"error": "Still starting up, please wait."}

        query_vec = self.embedder.embed_one(question)
        chunks = self.store.search(query_vec, top_k=5)

        full_text = ""
        for token in self.engine.ask(question, chunks):
            full_text += token
            self._push("onToken", token)
        self._push("onDone")

        citations = [
            {
                "index": i + 1,
                "doc_path": c["doc_path"],
                "doc_name": Path(c["doc_path"]).name,
                "page": c["page"],
                "page_width": c["page_width"],
                "page_height": c["page_height"],
                "bboxes": c["bboxes"],
            }
            for i, c in enumerate(chunks)
        ]
        return {"answer": full_text, "citations": citations}
