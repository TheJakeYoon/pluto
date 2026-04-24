"""File loaders for supported insertion formats.

Supported: pdf, txt, md, jpg, jpeg, png, heic, json, html/htm, doc, docx, xls, xlsx, ppt, pptx.

Strategy:
- Text-native formats (txt/md/json/html/csv) → decoded directly.
- Office formats → python-docx / python-pptx / openpyxl.
- PDF → PyMuPDF (fitz) for text. A helper also renders pages to PNG bytes so the
  vision module can run a multimodal LLM over them (for scans / image-heavy PDFs).
- Images → returned as PNG bytes + data URL for the vision module; caller should
  run ``vision.vision_ocr_image`` to turn them into text.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DocumentLoadResult:
    """Unified return type for any loader call."""

    text: str = ""
    page_images: List[bytes] = field(default_factory=list)  # PNG bytes per page/frame
    images: List[bytes] = field(default_factory=list)  # extra embedded images (raw PNG)
    metadata: dict = field(default_factory=dict)
    needs_vision: bool = False  # true when caller should run vision OCR to get text
    warnings: List[str] = field(default_factory=list)


# ---------- individual loaders ----------


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _load_txt_md(path: str) -> DocumentLoadResult:
    return DocumentLoadResult(text=_read_text(path), metadata={"loader": "text"})


def _load_json(path: str) -> DocumentLoadResult:
    raw = _read_text(path)
    try:
        parsed = json.loads(raw)
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        return DocumentLoadResult(text=pretty, metadata={"loader": "json"})
    except json.JSONDecodeError:
        return DocumentLoadResult(text=raw, metadata={"loader": "json-raw"}, warnings=["Invalid JSON; stored raw."])


def _load_html(path: str) -> DocumentLoadResult:
    raw = _read_text(path)
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        title = (soup.title.string or "").strip() if soup.title else ""
        return DocumentLoadResult(text=text, metadata={"loader": "html", "title": title})
    except Exception as e:  # pragma: no cover
        return DocumentLoadResult(text=raw, metadata={"loader": "html-raw"}, warnings=[f"bs4 failed: {e}"])


def extract_pdf_text(path: str) -> Tuple[str, dict]:
    """Return ``(text, meta)`` extracted from a PDF using PyMuPDF."""
    import fitz  # type: ignore

    parts: List[str] = []
    with fitz.open(path) as doc:
        meta = {
            "loader": "pdf",
            "page_count": doc.page_count,
            "pdf_title": (doc.metadata or {}).get("title") or "",
        }
        for i, page in enumerate(doc):
            txt = page.get_text("text") or ""
            if txt.strip():
                parts.append(f"[Page {i + 1}]\n{txt.strip()}")
    return ("\n\n".join(parts).strip(), meta)


def pdf_page_images(path: str, *, max_pages: int = 20, zoom: float = 1.8) -> List[bytes]:
    """Render up to ``max_pages`` PDF pages as PNG bytes (for vision OCR)."""
    import fitz  # type: ignore

    out: List[bytes] = []
    with fitz.open(path) as doc:
        pages = min(doc.page_count, max_pages)
        matrix = fitz.Matrix(zoom, zoom)
        for i in range(pages):
            pix = doc[i].get_pixmap(matrix=matrix, alpha=False)
            out.append(pix.tobytes("png"))
    return out


def _load_pdf(path: str) -> DocumentLoadResult:
    text, meta = extract_pdf_text(path)
    needs_vision = len(text.strip()) < 40  # heuristic: scanned / image-only PDF
    result = DocumentLoadResult(text=text, metadata=meta, needs_vision=needs_vision)
    if needs_vision:
        try:
            result.page_images = pdf_page_images(path)
        except Exception as e:
            result.warnings.append(f"Could not render PDF pages for vision: {e}")
    return result


def _load_docx(path: str) -> DocumentLoadResult:
    from docx import Document  # type: ignore

    doc = Document(path)
    parts: List[str] = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join((cell.text or "").strip() for cell in row.cells)
            if row_text.strip():
                parts.append(row_text)
    return DocumentLoadResult(text="\n".join(parts), metadata={"loader": "docx"})


def _load_doc_legacy(path: str) -> DocumentLoadResult:
    # .doc (binary) isn't well supported in pure Python. Try textract-free fallback: read as bytes
    # and look for ASCII runs. Users should convert to docx for best results.
    raw = open(path, "rb").read()
    text_chunks: List[str] = []
    current: List[int] = []
    for b in raw:
        if 32 <= b <= 126 or b in (9, 10, 13):
            current.append(b)
        else:
            if len(current) > 8:
                text_chunks.append(bytes(current).decode("ascii", errors="ignore"))
            current = []
    if len(current) > 8:
        text_chunks.append(bytes(current).decode("ascii", errors="ignore"))
    return DocumentLoadResult(
        text="\n".join(text_chunks),
        metadata={"loader": "doc-legacy"},
        warnings=["Legacy .doc: text extraction is best-effort. Convert to .docx for fidelity."],
    )


def _load_xlsx(path: str) -> DocumentLoadResult:
    from openpyxl import load_workbook  # type: ignore

    wb = load_workbook(filename=path, data_only=True, read_only=True)
    lines: List[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        lines.append(f"# Sheet: {sheet_name}")
        for row in ws.iter_rows(values_only=True):
            cells = ["" if c is None else str(c) for c in row]
            if any(cell.strip() for cell in cells):
                lines.append("\t".join(cells))
        lines.append("")
    return DocumentLoadResult(text="\n".join(lines), metadata={"loader": "xlsx"})


def _load_pptx(path: str) -> DocumentLoadResult:
    from pptx import Presentation  # type: ignore

    prs = Presentation(path)
    slides: List[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        bits: List[str] = [f"[Slide {i}]"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        bits.append(text)
            elif getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    row_text = "\t".join((cell.text or "").strip() for cell in row.cells)
                    if row_text.strip():
                        bits.append(row_text)
        slides.append("\n".join(bits))
    return DocumentLoadResult(text="\n\n".join(slides), metadata={"loader": "pptx"})


def _load_image(path: str) -> DocumentLoadResult:
    """Load image as PNG bytes and defer text extraction to the vision helper."""
    from PIL import Image  # type: ignore

    ext = os.path.splitext(path)[1].lower()
    if ext == ".heic":
        try:
            import pillow_heif  # type: ignore

            pillow_heif.register_heif_opener()
        except Exception:
            pass
    with Image.open(path) as img:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
    return DocumentLoadResult(
        text="",
        images=[png_bytes],
        metadata={"loader": "image", "format": ext.lstrip(".")},
        needs_vision=True,
    )


# ---------- dispatcher ----------


_EXT_LOADERS = {
    ".txt": _load_txt_md,
    ".md": _load_txt_md,
    ".json": _load_json,
    ".html": _load_html,
    ".htm": _load_html,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_doc_legacy,
    ".xlsx": _load_xlsx,
    ".xls": _load_xlsx,
    ".pptx": _load_pptx,
    ".jpg": _load_image,
    ".jpeg": _load_image,
    ".png": _load_image,
    ".heic": _load_image,
}


def load_file_to_text(path: str) -> DocumentLoadResult:
    """Dispatch to the correct loader based on the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _EXT_LOADERS:
        raise ValueError(
            f"Unsupported file type: {ext or '(none)'}. Supported: {', '.join(sorted(_EXT_LOADERS))}"
        )
    try:
        return _EXT_LOADERS[ext](path)
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency for {ext}: {e}. Install requirements.txt in the backend venv."
        ) from e


def image_to_data_url(png_bytes: bytes) -> str:
    """Encode PNG bytes as a data URL for multimodal LLM inputs."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"
