from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path):
    """Factory fixture: make_pdf(name, placements) -> Path to a generated PDF.

    `placements` is a list of (text, x, y) tuples inserted onto a single
    600x800 page, letting tests control exact block positions.
    """

    def _make(name: str, placements: list[tuple[str, float, float]]) -> Path:
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        for text, x, y in placements:
            page.insert_text((x, y), text, fontsize=12)
        path = tmp_path / name
        doc.save(path)
        doc.close()
        return path

    return _make
