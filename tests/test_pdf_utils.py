from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image as PILImage

from paper_orchestra.utils.common import read_json, read_text, write_json
from paper_orchestra.utils.pdf import (
    PdfError,
    load_paper,
    load_paper_bytes,
    page_count,
    pdf_page_images,
    pdf_to_grid_images,
)


@pytest.fixture
def pdf_path(tmp_path):
    doc = pymupdf.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}: " + "lorem ipsum " * 20)
    path = tmp_path / "paper.pdf"
    doc.save(path)
    return path


def test_text_extraction(pdf_path):
    assert page_count(pdf_path) == 5
    text = load_paper(pdf_path)
    assert "Page 1:" in text and "Page 5:" in text
    assert "Page 3:" not in load_paper(pdf_path, num_pages=2)
    assert load_paper_bytes(pdf_path.read_bytes()) == text


def test_unreadable_or_empty_pdf_raises_pdf_error(tmp_path):
    with pytest.raises(PdfError):
        load_paper_bytes(b"definitely not a pdf")
    empty = tmp_path / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(empty)
    with pytest.raises(PdfError, match="extracted only"):
        load_paper(empty)


def test_page_images_and_grids(pdf_path, tmp_path):
    images = pdf_page_images(pdf_path, max_pages=2, image_format="jpeg")
    assert len(images) == 2 and images[0].startswith(b"\xff\xd8")
    grids = pdf_to_grid_images(pdf_path, tmp_path / "shots", scale=1.0, max_side=900)
    assert [p.rsplit("/", 1)[1] for p in grids] == ["grid_1.png", "grid_2.png"]
    sheet = PILImage.open(io.BytesIO(Path(grids[0]).read_bytes()))
    assert max(sheet.size) <= 900


def test_file_helpers_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_text(tmp_path / "missing.md")
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")
    assert read_json(tmp_path / "missing.json", default=[]) == []
    write_json(tmp_path / "deep" / "x.json", {"a": 1})
    assert read_json(tmp_path / "deep" / "x.json") == {"a": 1}
    assert not list((tmp_path / "deep").glob(".*tmp"))
