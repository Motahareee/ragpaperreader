"""Step 1: establish baseline-zero numbers for the current system
(MiniLM embedder + brute-force cosine search) before comparing any
alternative encoders or retrieval methods against it.

Usage: python benchmark/run_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.harness import (  # noqa: E402
    measure_indexing_latency,
    measure_recall,
    model_file_size_mb,
    peak_rss_mb,
    save_report,
    sweep_search_latency,
)
from paperchat.embeddings import DEFAULT_EMBED_MODEL_PATH, Embedder  # noqa: E402
from paperchat.indexing import discover_pdfs  # noqa: E402
from paperchat.store import Store  # noqa: E402

TEST_PAPERS_DIR = REPO_ROOT / "test_papers"
GOLDEN_SET_PATH = REPO_ROOT / "tests" / "golden" / "qa_set.json"


def main():
    print("=== Baseline: MiniLM (GGUF, Q8_0) + brute-force cosine ===\n")

    print("Loading embedder...")
    embedder = Embedder()
    rss_after_model_load = peak_rss_mb()
    print(f"  peak RSS after model load: {rss_after_model_load:.1f} MB")
    print(f"  model file size: {model_file_size_mb(DEFAULT_EMBED_MODEL_PATH):.1f} MB\n")

    pdf_paths = discover_pdfs(TEST_PAPERS_DIR)
    print(f"Indexing {len(pdf_paths)} PDF(s) from scratch: {[p.name for p in pdf_paths]}")
    indexing_report = measure_indexing_latency(embedder, pdf_paths)
    for doc in indexing_report["per_doc"]:
        print(
            f"  {doc['doc']}: {doc['num_pages']} pages, {doc['num_chunks']} chunks, "
            f"parse={doc['parse_time_s']:.2f}s embed={doc['embed_time_s']:.2f}s "
            f"({doc['pages_per_sec']:.1f} pages/s)"
        )
    print(f"  peak RSS after indexing: {indexing_report['peak_rss_mb_after']:.1f} MB\n")

    # Build a store from the existing on-disk index (already incrementally
    # maintained across the session) for the recall measurement.
    store = Store(TEST_PAPERS_DIR)

    golden = json.loads(GOLDEN_SET_PATH.read_text())
    print(f"Measuring recall@5 on {len(golden['cases'])} golden questions...")
    recall_report = measure_recall(embedder, store, golden["cases"], top_k=5)
    print(f"  recall@5 = {recall_report['recall_at_k']:.2f} ({sum(c['hit'] for c in recall_report['per_case'])}/{recall_report['n_cases']})")
    avg_embed_ms = sum(c["embed_time_s"] for c in recall_report["per_case"]) / len(recall_report["per_case"]) * 1000
    avg_search_ms = sum(c["search_time_s"] for c in recall_report["per_case"]) / len(recall_report["per_case"]) * 1000
    print(f"  avg query embed time: {avg_embed_ms:.2f}ms, avg search time: {avg_search_ms:.3f}ms\n")

    print("Sweeping brute-force search latency vs. corpus size...")
    scalability_report = sweep_search_latency()
    for r in scalability_report["results"]:
        print(f"  n={r['n_vectors']:>7}: mean={r['mean_latency_ms']:.3f}ms p95={r['p95_latency_ms']:.3f}ms")

    report = {
        "variant": "minilm_cosine_baseline",
        "model_file_size_mb": model_file_size_mb(DEFAULT_EMBED_MODEL_PATH),
        "peak_rss_mb_after_model_load": rss_after_model_load,
        "indexing": indexing_report,
        "recall": recall_report,
        "scalability": scalability_report,
    }
    out_path = REPO_ROOT / "benchmark" / "results" / "baseline_minilm.json"
    save_report(report, out_path)
    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()
