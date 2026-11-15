# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PDF text extraction and page rendering (PyMuPDF with a pypdf fallback)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class PdfError(RuntimeError):
    """The PDF could not be read or contains no usable text."""


def _open(source: str | Path | bytes):
    import pymupdf

    if isinstance(source, (bytes, bytearray)):
        return pymupdf.open(stream=bytes(source), filetype="pdf")
    return pymupdf.open(str(source))


def page_count(source: str | Path | bytes) -> int:
    with _open(source) as doc:
        return doc.page_count


def _extract_text(source: str | Path | bytes, num_pages: int | None) -> str:
    try:
        with _open(source) as doc:
            pages = list(doc)[:num_pages] if num_pages else list(doc)
            return "".join(page.get_text() for page in pages)
    except Exception as exc:
        log.warning("PyMuPDF could not read the PDF (%s); falling back to pypdf", exc)

    from pypdf import PdfReader

    stream = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else str(source)
    try:
        reader = PdfReader(stream)
        pages = reader.pages[:num_pages] if num_pages else reader.pages
        return "".join((page.extract_text() or "") for page in pages)
    except Exception as exc:
        raise PdfError(f"could not read PDF: {exc}") from exc


def load_paper(pdf_path: str | Path, num_pages: int | None = None, min_size: int = 100) -> str:
    """Return the text of a PDF; raise :class:`PdfError` if there is too little."""
    text = _extract_text(pdf_path, num_pages)
    if len(text) < min_size:
        raise PdfError(f"{pdf_path}: extracted only {len(text)} characters of text")
    return text


def load_paper_bytes(data: bytes, num_pages: int | None = None, min_size: int = 100) -> str:
    """Like :func:`load_paper` for an in-memory PDF."""
    text = _extract_text(data, num_pages)
    if len(text) < min_size:
        raise PdfError(f"extracted only {len(text)} characters of text from the PDF")
    return text


def pdf_page_images(
    source: str | Path | bytes,
    *,
    scale: float = 1.5,
    max_pages: int | None = None,
    image_format: str = "png",
) -> list[bytes]:
    """Render each page to an encoded image (``png`` or ``jpeg``)."""
    import pymupdf

    matrix = pymupdf.Matrix(scale, scale)
    images: list[bytes] = []
    with _open(source) as doc:
        for index, page in enumerate(doc):
            if max_pages is not None and index >= max_pages:
                break
            images.append(page.get_pixmap(matrix=matrix, alpha=False).tobytes(image_format))
    return images


def pdf_to_grid_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    scale: float = 2.0,
    *,
    max_side: int | None = None,
) -> list[str]:
    """Write 2x2 contact sheets of the pages as ``grid_<n>.png``; return their paths.

    ``max_side`` downsizes each sheet so its longer side fits a model's image
    limit while staying legible.
    """
    from PIL import Image as PILImage

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = [
        PILImage.open(io.BytesIO(data)).convert("RGB")
        for data in pdf_page_images(pdf_path, scale=scale)
    ]
    paths: list[str] = []
    for start in range(0, len(pages), 4):
        chunk = pages[start : start + 4]
        cell_w = max(p.width for p in chunk)
        cell_h = max(p.height for p in chunk)
        sheet = PILImage.new("RGB", (cell_w * 2, cell_h * 2), "white")
        for offset, page in enumerate(chunk):
            sheet.paste(page, ((offset % 2) * cell_w, (offset // 2) * cell_h))
        if max_side and max(sheet.size) > max_side:
            sheet.thumbnail((max_side, max_side), PILImage.LANCZOS)
        path = out_dir / f"grid_{start // 4 + 1}.png"
        sheet.save(path, format="PNG", optimize=True)
        paths.append(str(path))
    return paths
