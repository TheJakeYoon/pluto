"""
Best-effort repair of malformed JSON in HTTP response bodies (escape sequences, fences, trailing commas).

Used by middleware before sending ``application/json`` responses. Does not apply to streaming bodies.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List

from starlette.responses import Response as StarletteResponse
from starlette.responses import StreamingResponse as StarletteStreamingResponse

_log = logging.getLogger("uvicorn.error")

# Avoid buffering huge JSON bodies in middleware (override with JSON_REPAIR_MAX_BYTES).
_DEFAULT_MAX = 6 * 1024 * 1024


def _max_bytes() -> int:
    import os

    try:
        return int(os.environ.get("JSON_REPAIR_MAX_BYTES", str(_DEFAULT_MAX)))
    except ValueError:
        return _DEFAULT_MAX


def _repair_invalid_json_escape_sequences(s: str) -> str:
    """Duplicate backslashes where JSON would otherwise mis-parse LaTeX or invalid escapes."""
    out: List[str] = []
    i = 0
    n = len(s)
    in_str = False

    def _backslashes_before(pos: int) -> int:
        c = 0
        p = pos - 1
        while p >= 0 and s[p] == "\\":
            c += 1
            p -= 1
        return c

    while i < n:
        ch = s[i]
        if not in_str:
            out.append(ch)
            if ch == '"' and _backslashes_before(i) % 2 == 0:
                in_str = True
            i += 1
            continue

        if ch == '"' and _backslashes_before(i) % 2 == 0:
            in_str = False
            out.append(ch)
            i += 1
            continue

        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "t" and i + 2 < n and s[i + 2].isalpha():
                out.append("\\")
                out.append("\\")
                i += 1
                continue
            if nxt in ('"', "\\", "/", "b", "f", "n", "r", "t"):
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            if nxt == "u" and i + 5 < n:
                hex4 = s[i + 2 : i + 6]
                if len(hex4) == 4 and re.fullmatch(r"[0-9a-fA-F]{4}", hex4):
                    out.append(s[i : i + 6])
                    i += 6
                    continue
            out.append("\\")
            out.append("\\")
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _strip_markdown_json_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _remove_trailing_commas(text: str) -> str:
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = _TRAILING_COMMA.sub(r"\1", cur)
    return cur


def _parse_json_with_repairs(raw: str) -> Optional[Any]:
    """Return parsed object or None if unrecoverable."""
    candidates = [
        raw,
        raw.lstrip("\ufeff").strip(),
        _strip_markdown_json_fence(raw.lstrip("\ufeff")),
    ]
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        for attempt in (
            lambda s: s,
            _remove_trailing_commas,
            _repair_invalid_json_escape_sequences,
            lambda s: _repair_invalid_json_escape_sequences(_remove_trailing_commas(s)),
            lambda s: _remove_trailing_commas(_repair_invalid_json_escape_sequences(s)),
        ):
            try:
                t = attempt(c)
                return json.loads(t)
            except json.JSONDecodeError:
                continue
    return None


def repair_json_response_body(body: bytes) -> tuple[bytes, bool]:
    """
    If ``body`` is UTF-8 JSON (possibly invalid), return ``(fixed_bytes, True)`` when repaired,
    or ``(original_bytes, False)`` when already valid or when repair failed.
    """
    if not body:
        return body, False
    if len(body) > _max_bytes():
        return body, False
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return body, False

    try:
        json.loads(text)
        return body, False
    except json.JSONDecodeError:
        pass

    obj = _parse_json_with_repairs(text)
    if obj is None:
        _log.warning(
            "JSON response repair: could not parse JSON (len=%s, preview=%r)",
            len(body),
            text[:240].replace("\n", " ") + ("…" if len(text) > 240 else ""),
        )
        return body, False
    try:
        fixed = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        _log.warning("JSON response repair: could not re-serialize: %s", e)
        return body, False
    out = fixed.encode("utf-8")
    _log.info(
        "JSON response repair: fixed invalid JSON in response (output_len=%s)",
        len(out),
    )
    return out, True


def apply_json_response_middleware(response: Any) -> Any:
    """
    If ``response`` is a non-streaming HTTP response with ``Content-Type: application/json``,
    repair invalid JSON in the body when possible. Returns a new ``Response`` when the body changes.
    """
    if isinstance(response, StarletteStreamingResponse):
        return response
    ct = (response.headers.get("content-type") or "").lower()
    if "application/json" not in ct:
        return response
    body = getattr(response, "body", None)
    if not isinstance(body, (bytes, memoryview)):
        return response
    raw = bytes(body)
    new_body, changed = repair_json_response_body(raw)
    if not changed:
        return response

    media = ct.split(";", 1)[0].strip() or "application/json"
    new_headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return StarletteResponse(
        content=new_body,
        status_code=response.status_code,
        headers=new_headers,
        media_type=media,
        background=getattr(response, "background", None),
    )
