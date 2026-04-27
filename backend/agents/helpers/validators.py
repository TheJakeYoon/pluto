"""Validate agent output formats and write them to disk.

Supports: markdown, html, json, pdf.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ValidationResult:
    ok: bool
    format: str
    errors: list
    sanitized: Optional[str] = None  # the content (possibly cleaned) ready to write


def _validate_markdown(content: str) -> ValidationResult:
    if not isinstance(content, str) or not content.strip():
        return ValidationResult(False, "markdown", ["Markdown content is empty."])
    return ValidationResult(True, "markdown", [], content)


def _validate_json(content: str) -> ValidationResult:
    if not isinstance(content, str) or not content.strip():
        return ValidationResult(False, "json", ["JSON content is empty."])
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return ValidationResult(False, "json", [f"Invalid JSON: {e.msg} at line {e.lineno} col {e.colno}"])
    pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
    return ValidationResult(True, "json", [], pretty)


_HTML_OPEN_TAGS = re.compile(r"<([a-zA-Z][a-zA-Z0-9-]*)[^>]*?>", re.DOTALL)
_HTML_CLOSE_TAGS = re.compile(r"</([a-zA-Z][a-zA-Z0-9-]*)\s*>", re.DOTALL)


def _validate_html(content: str) -> ValidationResult:
    if not isinstance(content, str) or not content.strip():
        return ValidationResult(False, "html", ["HTML content is empty."])
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(content, "html.parser")
        if not soup.find():
            return ValidationResult(False, "html", ["No HTML tags detected; content looks like plain text."])
        return ValidationResult(True, "html", [], str(soup))
    except Exception as e:
        return ValidationResult(False, "html", [f"HTML parse failed: {e}"])


def _validate_pdf_markdown(content: str) -> ValidationResult:
    """Validates the content intended for PDF generation (expects markdown source)."""
    if not isinstance(content, str) or not content.strip():
        return ValidationResult(False, "pdf", ["PDF source (markdown) is empty."])
    return ValidationResult(True, "pdf", [], content)


def validate_output(content: str, fmt: str) -> ValidationResult:
    fmt = (fmt or "").strip().lower()
    if fmt == "markdown" or fmt == "md":
        return _validate_markdown(content)
    if fmt == "json":
        return _validate_json(content)
    if fmt == "html":
        return _validate_html(content)
    if fmt == "pdf":
        return _validate_pdf_markdown(content)
    return ValidationResult(False, fmt, [f"Unsupported format: {fmt!r}. Use markdown, html, json, or pdf."])


# ---------- writers ----------


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "output"
    return name[:120]


def _markdown_to_pdf(md_text: str, out_path: str) -> None:
    """Render markdown → HTML → PDF via reportlab.

    Uses a minimal flowable pipeline so we avoid heavy deps like weasyprint.
    """
    import markdown as md_lib  # type: ignore
    from reportlab.lib.pagesizes import LETTER  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
    from reportlab.platypus import (  # type: ignore
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Preformatted,
        ListFlowable,
        ListItem,
    )
    from bs4 import BeautifulSoup  # type: ignore

    html_body = md_lib.markdown(md_text, extensions=["extra", "sane_lists", "fenced_code"])
    soup = BeautifulSoup(html_body, "html.parser")

    styles = getSampleStyleSheet()
    story = []
    for element in soup.children:
        name = getattr(element, "name", None)
        if not name:
            text = str(element).strip()
            if text:
                story.append(Paragraph(text, styles["BodyText"]))
            continue
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            style = styles["Heading" + str(min(level, 4))]
            story.append(Paragraph(element.get_text(strip=True), style))
        elif name == "p":
            inner_html = element.decode_contents()
            story.append(Paragraph(inner_html, styles["BodyText"]))
        elif name in ("ul", "ol"):
            items = [
                ListItem(Paragraph(li.decode_contents(), styles["BodyText"]))
                for li in element.find_all("li", recursive=False)
            ]
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet" if name == "ul" else "1",
                    leftIndent=18,
                )
            )
        elif name == "pre":
            code = element.get_text()
            story.append(Preformatted(code, styles["Code"]))
        elif name == "hr":
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(element.get_text(strip=True), styles["BodyText"]))
        story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(out_path, pagesize=LETTER, title="Generated by Education Agent")
    doc.build(story)


def write_output_file(
    content: str,
    fmt: str,
    *,
    out_dir: str,
    base_name: Optional[str] = None,
) -> str:
    """Persist validated content to ``out_dir``. Returns the file path."""
    os.makedirs(out_dir, exist_ok=True)
    fmt = fmt.strip().lower()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = _safe_filename(base_name or f"output-{ts}")
    ext = {"markdown": ".md", "md": ".md", "json": ".json", "html": ".html", "pdf": ".pdf"}[fmt if fmt != "md" else "markdown"]
    path = os.path.join(out_dir, f"{stem}{ext}")

    if fmt in ("markdown", "md"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    elif fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    elif fmt == "html":
        # If content is a fragment, wrap it in a minimal html shell.
        lower = content.lstrip().lower()
        if not lower.startswith("<!doctype") and not lower.startswith("<html"):
            wrapped = (
                "<!DOCTYPE html>\n<html lang=\"en\"><head>"
                "<meta charset=\"utf-8\"><title>Education Agent Output</title>"
                "</head><body>\n" + content + "\n</body></html>"
            )
        else:
            wrapped = content
        with open(path, "w", encoding="utf-8") as f:
            f.write(wrapped)
    elif fmt == "pdf":
        _markdown_to_pdf(content, path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return path
