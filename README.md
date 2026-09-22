# PaperChat

A local, private RAG chatbot for a folder of research papers. Drop `PaperChat.app` into a folder of PDFs, run it, and ask questions — answers cite the exact source passage and highlight it directly in the PDF.

No data ever leaves the machine: papers, questions, and answers are processed entirely locally. The only network call the app ever makes is a one-time download of the language model's weights on first launch (never your documents).

## Features

- **Drop-in-folder workflow** — the app indexes whatever folder it's run from (or the folder it's sitting in, for the packaged `.app`)
- **Fully local** — local embeddings, local vector search, local LLM; nothing about your papers or questions is ever sent anywhere
- **Click-to-highlight citations** — answers cite numbered excerpts inline (e.g. `[1]`); click one and the PDF viewer jumps to that page and highlights the exact passage
- **Related-work generator** — pick a topic and check off papers to generate a one-page related-work section, with the same clickable citations
- **Incremental indexing** — re-running only re-embeds new or changed PDFs
- **Single packaged app** — no Docker, no separate model server to manage

## Architecture

Everything runs in one Python process, wrapped in a native window (`pywebview`) with no HTTP server:

| Piece | How |
|---|---|
| PDF parsing | PyMuPDF, with a column-aware heuristic for academic two-column layout |
| Embeddings | A small GGUF model, bundled in the app, run via `llama.cpp` (no torch dependency) |
| Vector store | Plain numpy array + JSON metadata in a `.ragindex/` folder — no external database |
| Answering | A mid-size instruct model (Qwen2.5-7B), downloaded once and cached locally |
| PDF viewer | `pdf.js`, with citation bounding boxes converted to on-canvas highlight overlays |

See `blog-post-paperchat.html` for the fuller story of *why* these choices were made (existing tools evaluated, the macOS Finder working-directory gotcha, how retrieval quality gets tested).

## Quick start (dev mode)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_embed_model.py
python -m paperchat.main /path/to/a/folder/of/pdfs
```

First run downloads the ~4.5GB instruct model (progress shown in the app); every run after that is fully offline.

> **macOS note:** if `pip install` fails building `llama-cpp-python`, see `build/fix_llama_cpp_mac.sh` — it skips a bundled CLI-tools build step that trips up some toolchains.

## Building the macOS app

```bash
./build/build_mac.sh
```

Produces `build/dist/PaperChat.app`. Unsigned, so first launch needs right-click → Open (or `xattr -cr PaperChat.app`) to get past Gatekeeper.

Local builds have been unreliable across different Mac toolchains — `.github/workflows/build-mac.yml` builds it on a clean GitHub-hosted macOS runner instead; check the repo's **Actions** tab for a downloadable build if you'd rather skip building locally.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Includes a hand-verified retrieval-quality golden set (`tests/golden/qa_set.json`) in addition to standard unit tests — see the tests themselves for what's covered vs. known limitations (e.g. tiny embedding model, heuristic column detection).
