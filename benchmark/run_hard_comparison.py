"""Run the adversarial hard_qa_set.json against all three variants
(MiniLM+cosine, MiniLM+hybrid RRF, bge-small+cosine) for a real
accuracy comparison -- the original golden set was curated to only
include already-working questions, so it can't show accuracy deltas.

Usage: python benchmark/run_hard_comparison.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.harness import measure_recall, save_report  # noqa: E402
from benchmark.run_hybrid import BM25Okapi, hybrid_search, tokenize  # noqa: E402
from paperchat.embeddings import Embedder  # noqa: E402
from paperchat.indexing import discover_pdfs  # noqa: E402
from paperchat.store import Store, sync_folder  # noqa: E402

TEST_PAPERS_DIR = REPO_ROOT / "test_papers"
HARD_SET_PATH = REPO_ROOT / "benchmark" / "hard_qa_set.json"
BGE_MODEL_PATH = REPO_ROOT / "benchmark" / "models" / "bge-small-en-v1.5-q8_0.gguf"


def report_hits(label: str, per_case: list[dict]):
    hits = sum(c["hit"] for c in per_case)
    print(f"\n{label}: {hits}/{len(per_case)}")
    for c in per_case:
        marker = "OK  " if c["hit"] else "MISS"
        print(f"  [{marker}] {c['question'][:65]}")


def main():
    cases = json.loads(HARD_SET_PATH.read_text())["cases"]
    results = {}

    # --- Variant 1: MiniLM + cosine ---
    minilm = Embedder()
    minilm_store = Store(TEST_PAPERS_DIR)
    r1 = measure_recall(minilm, minilm_store, cases, top_k=5)
    report_hits("MiniLM + cosine", r1["per_case"])
    results["minilm_cosine"] = r1["recall_at_k"]

    # --- Variant 2: MiniLM + hybrid (cosine + BM25 via RRF) ---
    corpus = [tokenize(m["text"]) for m in minilm_store.metadata]
    bm25 = BM25Okapi(corpus)
    hits2 = []
    for case in cases:
        got = hybrid_search(minilm, minilm_store, bm25, case["question"])
        got_pairs = [(Path(g["doc_path"]).name, g["page"]) for g in got]
        hit = any(d == case["expected_doc"] and p == case["expected_page"] for d, p in got_pairs)
        hits2.append({"question": case["question"], "hit": hit})
    report_hits("MiniLM + hybrid RRF", hits2)
    results["minilm_hybrid"] = sum(c["hit"] for c in hits2) / len(hits2)

    # --- Variant 3: bge-small + cosine ---
    bge = Embedder(model_path=BGE_MODEL_PATH)
    bge_dir = REPO_ROOT / "benchmark" / "bge_test_papers"
    if bge_dir.exists():
        shutil.rmtree(bge_dir)
    bge_dir.mkdir(parents=True)
    for p in discover_pdfs(TEST_PAPERS_DIR):
        shutil.copy(p, bge_dir / p.name)
    bge_store = sync_folder(bge_dir, bge)
    r3 = measure_recall(bge, bge_store, cases, top_k=5)
    report_hits("bge-small + cosine", r3["per_case"])
    results["bge_cosine"] = r3["recall_at_k"]
    shutil.rmtree(bge_dir)

    print("\n=== Summary (hard set, n={}) ===".format(len(cases)))
    for name, recall in results.items():
        print(f"  {name:20s}: {recall:.2f}")

    save_report(
        {"hard_set_results": results, "n_cases": len(cases)},
        REPO_ROOT / "benchmark" / "results" / "hard_comparison.json",
    )


if __name__ == "__main__":
    main()
