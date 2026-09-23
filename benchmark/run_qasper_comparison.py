"""All 3 retrieval variants against the Qasper external validation sample:
MiniLM+cosine, bge-small+cosine, MiniLM+hybrid (cosine+BM25 via RRF).

Reuses the same 20 sampled (question, evidence) pairs and the same
real-PDF pipeline as run_qasper.py, and the same word-overlap hit
metric (exact text matching isn't meaningful across differing
chunk/parser boundaries).

Usage: python benchmark/run_qasper_comparison.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.harness import save_report  # noqa: E402
from benchmark.run_hybrid import BM25Okapi, reciprocal_rank_fusion, tokenize  # noqa: E402
from benchmark.run_qasper import PAPER_IDS, QASPER_DIR, sample_questions, word_overlap  # noqa: E402
from paperchat.embeddings import Embedder  # noqa: E402
from paperchat.indexing import discover_pdfs  # noqa: E402
from paperchat.store import Store, sync_folder  # noqa: E402

WORD_OVERLAP_THRESHOLD = 0.6
BGE_MODEL_PATH = REPO_ROOT / "benchmark" / "models" / "bge-small-en-v1.5-q8_0.gguf"


def score(cases, embedder, store, search_fn) -> dict:
    hits = 0
    per_case = []
    for case in cases:
        results = search_fn(embedder, store, case["question"])
        best_overlap = 0.0
        for r in results:
            if case["paper_id"] not in r["doc_path"]:
                continue
            best_overlap = max(best_overlap, word_overlap(case["evidence"], r["text"]))
        hit = best_overlap >= WORD_OVERLAP_THRESHOLD
        hits += hit
        per_case.append({"question": case["question"], "best_overlap": best_overlap, "hit": hit})
    return {"recall_at_5": hits / len(cases), "hits": hits, "n": len(cases), "per_case": per_case}


def cosine_search(embedder, store, question, top_k=5):
    return store.search(embedder.embed_one(question), top_k=top_k)


def main():
    print("Sampling Qasper questions...")
    cases = sample_questions()
    print(f"  {len(cases)} questions across {len(PAPER_IDS)} papers\n")

    results = {}

    # --- Variant 1: MiniLM + cosine ---
    print("=== MiniLM + cosine ===")
    minilm = Embedder()
    minilm_store = Store(QASPER_DIR)
    r1 = score(cases, minilm, minilm_store, cosine_search)
    print(f"  recall@5 = {r1['recall_at_5']:.2f} ({r1['hits']}/{r1['n']})")
    results["minilm_cosine"] = r1

    # --- Variant 2: MiniLM + hybrid (cosine + BM25 via RRF) ---
    print("\n=== MiniLM + hybrid RRF ===")
    corpus = [tokenize(m["text"]) for m in minilm_store.metadata]
    bm25 = BM25Okapi(corpus)

    def hybrid_search_fn(embedder, store, question, top_k=5):
        query_vec = embedder.embed_one(question)
        cosine_ranking = list(__import__("numpy").argsort(-(store.embeddings @ query_vec)))
        bm25_ranking = list(__import__("numpy").argsort(-bm25.get_scores(tokenize(question))))
        fused = reciprocal_rank_fusion([cosine_ranking, bm25_ranking])[:top_k]
        return [store.metadata[i] for i in fused]

    r2 = score(cases, minilm, minilm_store, hybrid_search_fn)
    print(f"  recall@5 = {r2['recall_at_5']:.2f} ({r2['hits']}/{r2['n']})")
    results["minilm_hybrid"] = r2

    # --- Variant 3: bge-small + cosine ---
    print("\n=== bge-small + cosine ===")
    bge = Embedder(model_path=BGE_MODEL_PATH)
    bge_dir = REPO_ROOT / "benchmark" / "bge_qasper_papers"
    if bge_dir.exists():
        shutil.rmtree(bge_dir)
    bge_dir.mkdir(parents=True)
    for p in discover_pdfs(QASPER_DIR):
        shutil.copy(p, bge_dir / p.name)
    bge_store = sync_folder(bge_dir, bge)
    r3 = score(cases, bge, bge_store, cosine_search)
    print(f"  recall@5 = {r3['recall_at_5']:.2f} ({r3['hits']}/{r3['n']})")
    results["bge_cosine"] = r3
    shutil.rmtree(bge_dir)

    print("\n=== Summary (Qasper external, n={}) ===".format(len(cases)))
    for name, r in results.items():
        print(f"  {name:16s}: {r['recall_at_5']:.2f} ({r['hits']}/{r['n']})")

    save_report(
        {"n_cases": len(cases), "results": {k: v["recall_at_5"] for k, v in results.items()}, "detail": results},
        REPO_ROOT / "benchmark" / "results" / "qasper_comparison.json",
    )


if __name__ == "__main__":
    main()
