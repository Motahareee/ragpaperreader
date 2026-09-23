"""External validation: sample real (question, evidence-paragraph) pairs
from Qasper (AllenAI's evidence-grounded QA-over-NLP-papers benchmark),
fetch the ACTUAL PDFs from arXiv (Qasper's paper `id` is an arXiv id),
run them through our real PDF pipeline (not Qasper's pre-parsed text),
and check whether our retrieval surfaces a chunk that overlaps with
Qasper's independently-annotated evidence paragraph.

Text won't match exactly (different parsers, different chunk boundaries),
so a hit is defined as substantial word-overlap between the evidence
paragraph and a retrieved chunk, not an exact substring match.

Usage: python benchmark/run_qasper.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset  # noqa: E402

from benchmark.harness import save_report  # noqa: E402
from paperchat.embeddings import Embedder  # noqa: E402
from paperchat.store import sync_folder  # noqa: E402

QASPER_DIR = REPO_ROOT / "benchmark" / "qasper_papers"
PAPER_IDS = [
    "1910.03814", "1911.03243", "1909.00694", "1908.06606", "2003.03612",
    "1809.10644", "1907.01413", "1901.02262", "1610.09516", "1911.03562",
]
QUESTIONS_PER_PAPER = 2
WORD_OVERLAP_THRESHOLD = 0.6


def normalize_words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def word_overlap(evidence: str, chunk_text: str) -> float:
    """Fraction of evidence's distinctive words found in chunk_text."""
    ev_words = normalize_words(evidence)
    if not ev_words:
        return 0.0
    chunk_words = normalize_words(chunk_text)
    return len(ev_words & chunk_words) / len(ev_words)


def sample_questions() -> list[dict]:
    ds = load_dataset("allenai/qasper", split="train", revision="refs/convert/parquet")
    by_id = {ex["id"]: ex for ex in ds}

    cases = []
    for paper_id in PAPER_IDS:
        ex = by_id[paper_id]
        qas = ex["qas"]
        picked = 0
        for i, q in enumerate(qas["question"]):
            if picked >= QUESTIONS_PER_PAPER:
                break
            for a in qas["answers"][i]["answer"]:
                if not a["unanswerable"] and len(a["evidence"]) == 1 and len(a["evidence"][0]) > 100:
                    cases.append({"paper_id": paper_id, "question": q, "evidence": a["evidence"][0]})
                    picked += 1
                    break
    return cases


def main():
    print("Sampling Qasper questions...")
    cases = sample_questions()
    print(f"  {len(cases)} questions across {len(PAPER_IDS)} papers\n")

    print("Indexing real arXiv PDFs through our own pipeline...")
    embedder = Embedder()
    store = sync_folder(QASPER_DIR, embedder)
    print(f"  {store.num_chunks} chunks indexed\n")

    hits = 0
    per_case = []
    for case in cases:
        query_vec = embedder.embed_one(case["question"])
        results = store.search(query_vec, top_k=5)
        best_overlap = 0.0
        best_hit = False
        for r in results:
            if case["paper_id"] not in r["doc_path"]:
                continue
            ov = word_overlap(case["evidence"], r["text"])
            best_overlap = max(best_overlap, ov)
            if ov >= WORD_OVERLAP_THRESHOLD:
                best_hit = True
        hits += best_hit
        per_case.append(
            {
                "paper_id": case["paper_id"],
                "question": case["question"],
                "best_overlap": best_overlap,
                "hit": best_hit,
            }
        )
        marker = "OK  " if best_hit else "MISS"
        print(f"  [{marker}] overlap={best_overlap:.2f} :: {case['question'][:65]}")

    recall = hits / len(cases)
    print(f"\nQasper-derived recall@5 (real PDFs, real pipeline): {recall:.2f} ({hits}/{len(cases)})")

    save_report(
        {
            "source": "Qasper (AllenAI) -- real arXiv PDFs re-parsed through our own pipeline",
            "n_papers": len(PAPER_IDS),
            "n_cases": len(cases),
            "recall_at_5": recall,
            "overlap_threshold": WORD_OVERLAP_THRESHOLD,
            "per_case": per_case,
        },
        REPO_ROOT / "benchmark" / "results" / "qasper_external.json",
    )


if __name__ == "__main__":
    main()
