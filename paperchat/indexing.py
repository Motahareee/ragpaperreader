"""PDF parsing and chunking.

Extracts text blocks (with page number + bounding box) from PDFs using
PyMuPDF, orders them for multi-column academic layouts, and groups them
into overlapping chunks suitable for embedding.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Kept comfortably under the bundled embedding model's 512-token trained
# context (a block can push a chunk over target before it's flushed, so
# leave headroom rather than targeting close to the limit).
CHUNK_WORD_TARGET = 250
CHUNK_WORD_OVERLAP = 40


@dataclass
class Block:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    page_width: float
    page_height: float


@dataclass
class Chunk:
    doc_id: str
    doc_path: str
    chunk_index: int
    page: int
    page_width: float
    page_height: float
    bboxes: list[tuple[float, float, float, float]]
    text: str


def file_hash(path: Path) -> str:
    """Cheap content fingerprint (size + mtime) used for incremental re-indexing."""
    stat = path.stat()
    key = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def doc_id_for(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _order_blocks_by_column(raw_blocks: list[tuple], page_width: float) -> list[tuple]:
    """Sort text blocks into reading order, handling common two-column layouts.

    Blocks whose left edge is left of the page midpoint are treated as the
    left column, the rest as the right column; each column is then read
    top-to-bottom. This is a heuristic, not a full layout analysis.

    Known edge case: PyMuPDF's block extraction merges text across the full
    page width into one block when both columns happen to share the exact
    same text baseline, which defeats splitting for that row. Rare in real
    papers (paragraph starts across columns almost never align that
    precisely) but worth knowing if a page mis-orders.
    """
    midpoint = page_width / 2
    left_col = [b for b in raw_blocks if b[0] < midpoint]
    right_col = [b for b in raw_blocks if b[0] >= midpoint]

    # If the split is very lopsided, the page probably isn't two-column —
    # fall back to plain top-to-bottom, left-to-right order.
    if len(left_col) < 2 or len(right_col) < 2:
        return sorted(raw_blocks, key=lambda b: (b[1], b[0]))

    left_col.sort(key=lambda b: b[1])
    right_col.sort(key=lambda b: b[1])
    return left_col + right_col


def extract_blocks(pdf_path: Path) -> list[Block]:
    blocks: list[Block] = []
    with pymupdf.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            raw_blocks = page.get_text("blocks")
            ordered = _order_blocks_by_column(raw_blocks, page.rect.width)
            for b in ordered:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                text = text.strip()
                if not text:
                    continue
                blocks.append(
                    Block(
                        page=page_num,
                        bbox=(x0, y0, x1, y1),
                        text=text,
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                    )
                )
    return blocks


def chunk_blocks(blocks: list[Block], doc_id: str, doc_path: str) -> list[Chunk]:
    """Group blocks into ~CHUNK_WORD_TARGET-word chunks with a small overlap.

    A chunk never spans across a page boundary, so every chunk maps
    cleanly to a single page for highlighting.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)

    for page_num in sorted(by_page):
        page_blocks = by_page[page_num]
        current_blocks: list[Block] = []
        current_words = 0

        def flush():
            nonlocal chunk_index, current_blocks, current_words
            if not current_blocks:
                return
            text = "\n".join(b.text for b in current_blocks)
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    doc_path=doc_path,
                    chunk_index=chunk_index,
                    page=page_num,
                    page_width=current_blocks[0].page_width,
                    page_height=current_blocks[0].page_height,
                    bboxes=[b.bbox for b in current_blocks],
                    text=text,
                )
            )
            chunk_index += 1

        i = 0
        while i < len(page_blocks):
            b = page_blocks[i]
            current_blocks.append(b)
            current_words += len(b.text.split())
            if current_words >= CHUNK_WORD_TARGET:
                flush()
                # keep the tail of the current chunk as overlap for the next one
                overlap_blocks: list[Block] = []
                overlap_words = 0
                for ob in reversed(current_blocks):
                    overlap_words += len(ob.text.split())
                    overlap_blocks.insert(0, ob)
                    if overlap_words >= CHUNK_WORD_OVERLAP:
                        break
                current_blocks = overlap_blocks
                current_words = overlap_words
            i += 1
        flush()

    return chunks


def parse_pdf(pdf_path: Path) -> list[Chunk]:
    doc_id = doc_id_for(pdf_path)
    blocks = extract_blocks(pdf_path)
    return chunk_blocks(blocks, doc_id=doc_id, doc_path=str(pdf_path.resolve()))


def discover_pdfs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.pdf") if p.is_file())
