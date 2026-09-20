"""Retrieval regression test against a hand-verified golden Q&A set.

Requires the bundled embedding model and the sample PDF to be present
locally (both are gitignored / fetched separately, so this is skipped
rather than failed when they're missing -- see scripts/fetch_embed_model.py
and tests/golden/README below).
"""
import json
from pathlib import Path

import pytest

from paperchat.embeddings import DEFAULT_EMBED_MODEL_PATH, Embedder
from paperchat.store import sync_folder

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden" / "qa_set.json"
TEST_PAPERS_DIR = REPO_ROOT / "test_papers"

pytestmark = pytest.mark.skipif(
    not DEFAULT_EMBED_MODEL_PATH.exists(),
    reason=(
        "Embedding model not found -- run `python scripts/fetch_embed_model.py` "
        "before running retrieval quality tests."
    ),
)


@pytest.fixture(scope="module")
def indexed_store():
    if not any(TEST_PAPERS_DIR.glob("*.pdf")):
        pytest.skip(f"No sample PDFs in {TEST_PAPERS_DIR} to test retrieval against.")
    embedder = Embedder()
    store = sync_folder(TEST_PAPERS_DIR, embedder)
    return embedder, store


def _load_cases():
    """Returns [] (collecting zero tests, not an error) if the golden set's
    source PDF isn't present -- actual skip-with-reason happens via the
    `indexed_store` fixture for tests that use it at runtime."""
    if not GOLDEN_SET_PATH.exists():
        return []
    data = json.loads(GOLDEN_SET_PATH.read_text())
    pdf_path = TEST_PAPERS_DIR / data["source_pdf"]
    if not pdf_path.exists():
        return []
    return data["cases"]


@pytest.mark.parametrize("case", _load_cases())
def test_expected_page_is_in_top_5_results(indexed_store, case):
    embedder, store = indexed_store
    query_vec = embedder.embed_one(case["question"])
    results = store.search(query_vec, top_k=5)
    pages = [r["page"] for r in results]
    assert case["expected_page"] in pages, (
        f"Expected page {case['expected_page']} not in top-5 {pages} "
        f"for question: {case['question']!r}"
    )


def test_recall_at_5_meets_baseline(indexed_store):
    """Aggregate recall across the whole golden set shouldn't regress below
    what was verified when the set was built (currently 8/8)."""
    embedder, store = indexed_store
    cases = _load_cases()
    if not cases:
        pytest.skip("Golden set or its source PDF not available.")
    hits = 0
    for case in cases:
        query_vec = embedder.embed_one(case["question"])
        pages = [r["page"] for r in store.search(query_vec, top_k=5)]
        if case["expected_page"] in pages:
            hits += 1
    recall = hits / len(cases)
    assert recall >= 0.85, f"recall@5 dropped to {recall:.2f} ({hits}/{len(cases)})"
