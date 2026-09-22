# Handoff: PaperChat session summary + first prompt for the new photo-search project

## Summary of the PaperChat session

Built PaperChat (github.com/Motahareee/ragpaperreader) — a fully local, private RAG chatbot for a folder of research papers:

- **Architecture:** PyMuPDF for PDF parsing (column-aware chunking), a bundled GGUF MiniLM model for embeddings (via llama.cpp, no torch), a plain numpy+JSON vector store (no database), Qwen2.5-7B-Instruct as the answering model (auto-downloaded once, cached, fully offline after), pywebview + pdf.js for the GUI with click-to-highlight citations, packaged as a macOS .app via PyInstaller.
- **Key lessons hit along the way:** local llama-cpp-python builds were unreliable across Mac toolchains, so the .app gets built via GitHub Actions on a clean macos-14 runner instead of locally. PyInstaller silently drops llama-cpp-python's .dylib (loaded dynamically via ctypes) unless you explicitly `collect_dynamic_libs("llama_cpp")` in the spec. Small local LLMs will hallucinate citation numbers by reproducing a source PDF's own inline reference markers unless you strip those out of the excerpts first.
- **Shipped features:** working Q&A with citations, a full UI redesign (markdown rendering, inline citation badges, typing indicator, zoom controls), and a related-work section generator (topic + selected papers -> streamed one-page synthesis with citations, using per-paper retrieval so every selected paper is guaranteed coverage).
- **Testing:** pytest suite including a hand-verified retrieval golden set, with real, documented limitations of the tiny embedding model rather than a claim of perfect retrieval.

Now starting a new, separate project: local macOS photo search by time/place/people (free via osxphotos reading Photos.app's own metadata) and activity/content (needs a CLIP-style embedding model, new work). Set up in its own repo/folder (photosearchrag), with a Docker container so Claude Code's file access is scoped to just that folder.

---

## First prompt for the new project

Copy everything between the lines below and paste it as your first message once `claude` is running inside the new container.

---

Starting a new local, private macOS project: photo search by time, place,
people, and activity. Goal: search personal photos like "photos from the
beach last summer with Sarah" -- combining structured metadata (date, GPS
location, person names) with semantic visual search over photo content.

Plan so far:
- Use `osxphotos` (Python) to read Apple Photos.app's own library data
  directly: timestamps, reverse-geocoded place names, and person names
  from Photos' built-in face recognition. This needs zero ML -- it's
  already computed by Photos.app.
- For "activity" / visual-content search (the genuinely new capability),
  embed photos with a CLIP-style image encoder and embed text queries the
  same way, then cosine-similarity search -- same pattern as below.

This is a sibling project to github.com/Motahareee/ragpaperreader
(PaperChat, a local RAG chatbot for PDFs). Reuse these patterns from it,
they're proven to work:
1. Local vector store = plain numpy array + JSON metadata sidecar, no
   external database. Fine at this scale, avoids fragile dependencies
   (e.g. onnxruntime) that are painful to bundle with PyInstaller.
2. GUI = pywebview + local HTML/JS, no HTTP server -- a Python object is
   exposed to JS via pywebview's JS-API bridge.
3. Package with PyInstaller, but build the .app via GitHub Actions on a
   macos-14 runner instead of locally -- local builds kept hitting
   llama-cpp-python C++ toolchain failures that a clean CI runner avoids.
4. If using llama.cpp for anything (e.g. a GGUF CLIP variant), the spec
   file needs `binaries=collect_dynamic_libs("llama_cpp")` -- the .dylib
   is loaded dynamically via ctypes and PyInstaller's static analysis
   misses it otherwise. This caused a real "app opens then immediately
   crashes" bug last time.
5. If prompting a local LLM to cite numbered sources, explicitly warn it
   that source text may contain its own inline reference numbers (not
   your citation scheme) -- small local models will otherwise hallucinate
   citations by reproducing them.

First step: let's scope out what `osxphotos` actually exposes (run it
against my real Photos library and show me sample output for a few
photos -- dates, GPS/place, person tags, keywords) before we design
anything further.

---
