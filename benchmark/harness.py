"""Reusable measurement functions for the systems-evaluation benchmark.

Not part of the shipped app -- this is evaluation/instrumentation code for
comparing encoder and retrieval-method variants on accuracy, latency, and
memory. Kept separate from paperchat/ deliberately.

Hardware caveat: this runs in an emulated Linux ARM64 sandbox, not the
target Apple Silicon Mac (no Metal acceleration available here). Numbers
from this harness are trustworthy for *relative* comparisons between
variants measured here, but won't match absolute latency on real hardware.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Callable

import numpy as np


def peak_rss_mb() -> float:
    """Peak RSS (resident set size) of this process, in MB, since it started.

    ru_maxrss is cumulative/monotonic (never decreases), so calling this
    before and after a phase and reporting the "after" value gives peak
    memory *up to and including* that phase -- which is realistic, since
    e.g. the embedding model is still loaded in memory during generation.
    Linux reports ru_maxrss in KB; macOS reports it in bytes (not handled
    here since this harness targets the Linux sandbox).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def time_it(fn: Callable) -> tuple:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def model_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def measure_recall(embedder, store, golden_cases: list[dict], top_k: int = 5) -> dict:
    """Recall@top_k against a golden (question, expected_doc, expected_page) set.

    Matches on (doc, page), not page alone -- a store spanning more than
    one document could otherwise register a false-positive hit purely from
    a same-numbered page in the wrong paper.
    """
    hits = 0
    per_case = []
    for case in golden_cases:
        query_vec, embed_time = time_it(lambda: embedder.embed_one(case["question"]))
        results, search_time = time_it(lambda: store.search(query_vec, top_k=top_k))
        got = [(Path(r["doc_path"]).name, r["page"]) for r in results]
        expected_doc = case.get("expected_doc")
        if expected_doc:
            hit = any(doc == expected_doc and page == case["expected_page"] for doc, page in got)
        else:
            hit = any(page == case["expected_page"] for _, page in got)
        hits += hit
        per_case.append(
            {
                "question": case["question"],
                "expected_page": case["expected_page"],
                "expected_doc": expected_doc,
                "got_pages": [p for _, p in got],
                "got": got,
                "hit": hit,
                "embed_time_s": embed_time,
                "search_time_s": search_time,
            }
        )
    return {
        "recall_at_k": hits / len(golden_cases) if golden_cases else 0.0,
        "k": top_k,
        "n_cases": len(golden_cases),
        "per_case": per_case,
    }


def measure_indexing_latency(embedder, pdf_paths: list[Path]) -> dict:
    """Parse+chunk+embed each PDF from scratch (bypassing the incremental
    on-disk cache) and report per-stage timing."""
    from paperchat.indexing import parse_pdf

    results = []
    rss_before = peak_rss_mb()
    for pdf_path in pdf_paths:
        chunks, parse_time = time_it(lambda: parse_pdf(pdf_path))
        vectors, embed_time = time_it(lambda: embedder.embed([c.text for c in chunks]))
        import pymupdf

        num_pages = len(pymupdf.open(pdf_path))
        results.append(
            {
                "doc": pdf_path.name,
                "num_pages": num_pages,
                "num_chunks": len(chunks),
                "parse_time_s": parse_time,
                "embed_time_s": embed_time,
                "pages_per_sec": num_pages / (parse_time + embed_time) if (parse_time + embed_time) else None,
            }
        )
    return {
        "per_doc": results,
        "total_pages": sum(r["num_pages"] for r in results),
        "total_time_s": sum(r["parse_time_s"] + r["embed_time_s"] for r in results),
        "peak_rss_mb_after": peak_rss_mb(),
        "peak_rss_mb_before": rss_before,
    }


def sweep_search_latency(
    dim: int = 384, sizes: list[int] | None = None, n_queries: int = 20
) -> dict:
    """Time brute-force numpy cosine search (what store.py actually does)
    against synthetic random unit vectors at increasing corpus sizes, to
    find where a real ANN index would start to matter."""
    sizes = sizes or [10, 100, 1_000, 10_000, 100_000, 500_000]
    rng = np.random.default_rng(0)
    results = []
    for n in sizes:
        matrix = rng.normal(size=(n, dim)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        queries = rng.normal(size=(n_queries, dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        times = []
        for q in queries:
            _, elapsed = time_it(lambda: np.argsort(-(matrix @ q))[:5])
            times.append(elapsed)
        results.append(
            {
                "n_vectors": n,
                "mean_latency_ms": float(np.mean(times) * 1000),
                "p95_latency_ms": float(np.percentile(times, 95) * 1000),
            }
        )
    return {"dim": dim, "results": results}


def save_report(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
