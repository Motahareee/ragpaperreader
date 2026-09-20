import numpy as np

from paperchat.indexing import Chunk
from paperchat.store import Store


def _chunk(doc_path, idx, text="text"):
    return Chunk(
        doc_id="d",
        doc_path=doc_path,
        chunk_index=idx,
        page=idx,
        page_width=600,
        page_height=800,
        bboxes=[(0, 0, 10, 10)],
        text=text,
    )


def test_search_returns_nearest_by_cosine_similarity(tmp_path):
    store = Store(tmp_path)
    chunks = [_chunk("a.pdf", 0), _chunk("a.pdf", 1)]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.add_chunks(chunks, vectors)

    results = store.search(np.array([0.9, 0.1], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_index"] == 0


def test_remove_document_drops_only_its_chunks(tmp_path):
    store = Store(tmp_path)
    store.add_chunks(
        [_chunk("a.pdf", 0), _chunk("b.pdf", 0)],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    store.remove_document("a.pdf")
    assert store.num_chunks == 1
    assert store.metadata[0]["doc_path"] == "b.pdf"
    assert store.embeddings.shape[0] == 1


def test_needs_reindex_skips_unchanged_files(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 content")
    store = Store(tmp_path)
    store.mark_indexed(pdf)
    assert store.needs_reindex([pdf]) == []

    pdf.write_bytes(b"%PDF-1.4 changed content, different size")
    assert store.needs_reindex([pdf]) == [pdf]


def test_prune_deleted_removes_manifest_and_chunks(tmp_path):
    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"a")
    store = Store(tmp_path)
    store.add_chunks([_chunk(str(pdf_a.resolve()), 0)], np.array([[1.0, 0.0]], dtype=np.float32))
    store.mark_indexed(pdf_a)

    store.prune_deleted([])  # pdf_a no longer present on disk

    assert store.num_chunks == 0
    assert store.manifest == {}


def test_save_and_reload_roundtrip(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"a")
    store = Store(tmp_path)
    store.add_chunks([_chunk("a.pdf", 0)], np.array([[1.0, 0.0]], dtype=np.float32))
    store.mark_indexed(pdf)
    store.save()

    reloaded = Store(tmp_path)
    assert reloaded.num_chunks == 1
    assert reloaded.metadata[0]["doc_path"] == "a.pdf"
    np.testing.assert_array_equal(reloaded.embeddings, store.embeddings)
