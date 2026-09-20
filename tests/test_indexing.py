from paperchat.indexing import (
    Block,
    chunk_blocks,
    discover_pdfs,
    extract_blocks,
    file_hash,
)


def test_extract_blocks_reads_simple_text(make_pdf):
    path = make_pdf("simple.pdf", [("Hello world", 50, 50)])
    blocks = extract_blocks(path)
    assert len(blocks) == 1
    assert blocks[0].page == 0
    assert "Hello world" in blocks[0].text
    assert blocks[0].page_width == 600
    assert blocks[0].page_height == 800


def test_two_column_blocks_read_in_column_major_order(make_pdf):
    # Two blocks per side is the minimum for the column heuristic to kick in.
    # Note: rows are given a slight vertical offset between columns -- PyMuPDF's
    # block extraction merges same-row text across the whole page width into a
    # single block when both sides sit on the exact same baseline, which would
    # defeat column splitting entirely. Real papers rarely align paragraph
    # starts across columns to the same pixel row, but it's a known edge case.
    path = make_pdf(
        "twocol.pdf",
        [
            ("LEFT-TOP", 50, 50),
            ("RIGHT-TOP", 350, 65),
            ("LEFT-BOTTOM", 50, 400),
            ("RIGHT-BOTTOM", 350, 415),
        ],
    )
    blocks = extract_blocks(path)
    texts = [b.text.strip() for b in blocks]
    assert texts == ["LEFT-TOP", "LEFT-BOTTOM", "RIGHT-TOP", "RIGHT-BOTTOM"]


def test_single_column_falls_back_to_top_to_bottom(make_pdf):
    # Only one block on the "right" side -> below the 2-per-column threshold,
    # so it should fall back to plain top-to-bottom order, not column split.
    path = make_pdf(
        "onecol.pdf",
        [
            ("FIRST", 50, 50),
            ("SECOND", 50, 150),
            ("THIRD", 350, 250),
        ],
    )
    blocks = extract_blocks(path)
    texts = [b.text.strip() for b in blocks]
    assert texts == ["FIRST", "SECOND", "THIRD"]


def _block(page, text, y=0.0, page_width=600, page_height=800):
    return Block(page=page, bbox=(0, y, 100, y + 10), text=text, page_width=page_width, page_height=page_height)


def test_chunk_blocks_groups_to_word_target():
    words_per_block = 50
    block_text = " ".join(f"w{i}" for i in range(words_per_block))
    blocks = [_block(0, block_text, y=i * 10) for i in range(10)]  # 500 words total

    chunks = chunk_blocks(blocks, doc_id="doc1", doc_path="/tmp/doc1.pdf")

    assert len(chunks) >= 2
    # every chunk but possibly the last should have reached the word target
    for c in chunks[:-1]:
        assert len(c.text.split()) >= 250  # CHUNK_WORD_TARGET
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.page_width == 600 and c.page_height == 800 for c in chunks)


def test_chunks_never_cross_a_page_boundary():
    blocks = [
        _block(0, "short text on page one", y=0),
        _block(1, "different text on page two", y=0),
    ]
    chunks = chunk_blocks(blocks, doc_id="d", doc_path="/tmp/d.pdf")
    assert {c.page for c in chunks} == {0, 1}
    assert len(chunks) == 2


def test_file_hash_reflects_size_changes(tmp_path):
    p = tmp_path / "f.pdf"
    p.write_bytes(b"hello")
    h1 = file_hash(p)
    p.write_bytes(b"hello, much longer content now")
    h2 = file_hash(p)
    assert h1 != h2


def test_file_hash_stable_for_unchanged_file(tmp_path):
    p = tmp_path / "f.pdf"
    p.write_bytes(b"hello")
    assert file_hash(p) == file_hash(p)


def test_discover_pdfs_only_lists_pdfs(tmp_path):
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.txt").write_text("hi")
    result = discover_pdfs(tmp_path)
    assert [p.name for p in result] == ["a.pdf", "b.pdf"]
