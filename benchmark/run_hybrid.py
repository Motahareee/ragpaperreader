"""Step 3: hybrid retrieval (dense cosine + sparse BM25 keyword matching,
combined via Reciprocal Rank Fusion) vs. pure-cosine baseline, on the same
MiniLM-embedded store and golden set used in step 1.

Usage: python benchmark/run_hybrid.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rank_bm25 import BM25Okapi  # noqa: E402

from benchmark.harness import time_it  # noqa: E402
from paperchat.embeddings import Embedder  # noqa: E402
from paperchat.store import Store  # noqa: E402

TEST_PAPERS_DIR = REPO_ROOT / "test_papers"
GOLDEN_SET_PATH = REPO_ROOT / "tests" / "golden" / "qa_set.json"

RRF_K = 60  # standard reciprocal-rank-fusion constant


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """rankings: list of ranked index-lists (best first). Returns fused ranking."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: -scores[i])


def hybrid_search(embedder, store, bm25, question: str, top_k: int = 5) -> list[dict]:
    query_vec = embedder.embed_one(question)
    cosine_scores = store.embeddings @ query_vec
    cosine_ranking = list(np_argsort_desc(cosine_scores))

    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_ranking = list(np_argsort_desc(bm25_scores))

    fused = reciprocal_rank_fusion([cosine_ranking, bm25_ranking])[:top_k]
    return [store.metadata[i] for i in fused]


def np_argsort_desc(scores):
    import numpy as np

    return np.argsort(-scores)


def main():
    print("=== Variant: hybrid (cosine + BM25 via RRF) on MiniLM embeddings ===\n")

    embedder = Embedder()
    store = Store(TEST_PAPERS_DIR)
    print(f"Loaded store: {store.num_chunks} chunks\n")

    print("Building BM25 index over chunk texts...")
    corpus = [tokenize(m["text"]) for m in store.metadata]
    bm25 = BM25Okapi(corpus)

    golden = json.loads(GOLDEN_SET_PATH.read_text())
    hits = 0
    per_case = []
    print(f"Measuring hybrid recall@5 on {len(golden['cases'])} golden questions...")
    for case in golden["cases"]:
        results, elapsed = time_it(lambda: hybrid_search(embedder, store, bm25, case["question"]))
        pages = [r["page"] for r in results]
        hit = case["expected_page"] in pages
        hits += hit
        per_case.append({"question": case["question"], "expected_page": case["expected_page"], "got_pages": pages, "hit": hit, "time_s": elapsed})
        marker = "OK  " if hit else "MISS"
        print(f"  [{marker}] expected={case['expected_page']} got={pages} :: {case['question'][:60]}")

    recall = hits / len(golden["cases"])
    avg_time_ms = sum(c["time_s"] for c in per_case) / len(per_case) * 1000
    print(f"\n  hybrid recall@5 = {recall:.2f} ({hits}/{len(golden['cases'])})")
    print(f"  avg query time (embed+bm25+fuse): {avg_time_ms:.2f}ms")

    report = {
        "variant": "minilm_hybrid_rrf",
        "recall_at_k": recall,
        "n_cases": len(golden["cases"]),
        "per_case": per_case,
        "avg_query_time_ms": avg_time_ms,
    }
    from benchmark.harness import save_report

    out_path = REPO_ROOT / "benchmark" / "results" / "hybrid_rrf.json"
    save_report(report, out_path)
    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()
