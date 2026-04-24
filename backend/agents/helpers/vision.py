"""Vision OCR via a multimodal Ollama chat model.

Used by the insertion agent to turn image bytes (or rendered PDF pages) into
text that can be embedded into ChromaDB.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from .loaders import image_to_data_url

VISION_PROMPT = (
    "Transcribe ALL text visible in this image, in reading order. "
    "Preserve headings, lists and tables (tabs separate columns). "
    "If the image is a diagram or chart, briefly describe it after the transcription "
    "under a 'Description:' section. Output plain text only."
)


def _invoke_vision(model: str, png_bytes_list: List[bytes], *, prompt: str) -> str:
    llm = ChatOllama(model=model, temperature=0, keep_alive=-1)
    content = [{"type": "text", "text": prompt}]
    for b in png_bytes_list:
        content.append({"type": "image_url", "image_url": image_to_data_url(b)})
    msg = HumanMessage(content=content)
    result = llm.invoke([msg])
    text = getattr(result, "content", "") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
    return str(text).strip()


def vision_ocr_image(png_bytes: bytes, *, model: str, prompt: str = VISION_PROMPT) -> str:
    """Return transcribed text for a single image using a vision LLM."""
    return _invoke_vision(model, [png_bytes], prompt=prompt)


def vision_ocr_pdf_pages(
    pages_png: List[bytes],
    *,
    model: str,
    per_page: bool = True,
    prompt: str = VISION_PROMPT,
) -> str:
    """Return transcribed text for a list of PDF page images.

    ``per_page=True`` does one vision call per page (more reliable for long PDFs)
    and prefixes each page with ``[Page N]``.
    """
    if not pages_png:
        return ""
    if not per_page:
        return _invoke_vision(model, pages_png, prompt=prompt)
    parts: List[str] = []
    for i, png in enumerate(pages_png, start=1):
        try:
            text = _invoke_vision(model, [png], prompt=prompt)
        except Exception as e:
            text = f"(vision OCR failed: {e})"
        parts.append(f"[Page {i}]\n{text}")
    return "\n\n".join(parts).strip()
