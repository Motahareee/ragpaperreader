"""Step 2: same harness, swapped to bge-small-en-v1.5 (GGUF, Q8_0) instead
of MiniLM, for an apples-to-apples encoder comparison.

Usage: python benchmark/run_bge.py
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
from paperchat.embeddings import Embedder  # noqa: E402
from paperchat.indexing import discover_pdfs  # noqa: E402
from paperchat.store import Store  # noqa: E402

TEST_PAPERS_DIR = REPO_ROOT / "test_papers"
GOLDEN_SET_PATH = REPO_ROOT / "tests" / "golden" / "qa_set.json"
BGE_MODEL_PATH = REPO_ROOT / "benchmark" / "models" / "bge-small-en-v1.5-q8_0.gguf"


def main():
    print("=== Variant: bge-small-en-v1.5 (GGUF, Q8_0) + brute-force cosine ===\n")

    print("Loading embedder...")
    embedder = Embedder(model_path=BGE_MODEL_PATH)
    rss_after_model_load = peak_rss_mb()
    print(f"  peak RSS after model load: {rss_after_model_load:.1f} MB")
    print(f"  model file size: {model_file_size_mb(BGE_MODEL_PATH):.1f} MB\n")

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

    # Build a bge-embedded store from scratch (separate index dir so it
    # doesn't clobber the MiniLM baseline's .ragindex).
    import shutil

    bge_papers_dir = REPO_ROOT / "benchmark" / "bge_test_papers"
    if bge_papers_dir.exists():
        shutil.rmtree(bge_papers_dir)
    bge_papers_dir.mkdir(parents=True)
    for p in pdf_paths:
        shutil.copy(p, bge_papers_dir / p.name)

    from paperchat.store import sync_folder

    store = sync_folder(bge_papers_dir, embedder)

    golden = json.loads(GOLDEN_SET_PATH.read_text())
    print(f"Measuring recall@5 on {len(golden['cases'])} golden questions...")
    recall_report = measure_recall(embedder, store, golden["cases"], top_k=5)
    print(f"  recall@5 = {recall_report['recall_at_k']:.2f} ({sum(c['hit'] for c in recall_report['per_case'])}/{recall_report['n_cases']})")
    avg_embed_ms = sum(c["embed_time_s"] for c in recall_report["per_case"]) / len(recall_report["per_case"]) * 1000
    avg_search_ms = sum(c["search_time_s"] for c in recall_report["per_case"]) / len(recall_report["per_case"]) * 1000
    print(f"  avg query embed time: {avg_embed_ms:.2f}ms, avg search time: {avg_search_ms:.3f}ms\n")

    for case in recall_report["per_case"]:
        marker = "OK  " if case["hit"] else "MISS"
        print(f"  [{marker}] expected={case['expected_page']} got={case['got_pages']} :: {case['question'][:60]}")

    report = {
        "variant": "bge_small_cosine",
        "model_file_size_mb": model_file_size_mb(BGE_MODEL_PATH),
        "peak_rss_mb_after_model_load": rss_after_model_load,
        "indexing": indexing_report,
        "recall": recall_report,
    }
    out_path = REPO_ROOT / "benchmark" / "results" / "bge_small.json"
    save_report(report, out_path)
    print(f"\nSaved full report to {out_path}")

    shutil.rmtree(bge_papers_dir)


if __name__ == "__main__":
    main()
