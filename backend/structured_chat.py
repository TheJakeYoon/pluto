"""
Structured assistant replies via forced tool calls.

Cloud APIs parse tool/function arguments as JSON, which avoids asking the model to
emit raw JSON in assistant text (a common source of invalid escapes and LaTeX
backslash corruption). Math is isolated in typed segments so TeX stays in
dedicated JSON string values with normal JSON escaping.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import httpx

TOOL_NAME = "deliver_chat_response"

# Shared JSON Schema for tool parameters (OpenAI `parameters`, Anthropic `input_schema`).
DELIVER_CHAT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "description": (
                "Ordered parts of the reply. Use markdown for prose; use math_inline / "
                "math_display for any TeX (including content that contains backslashes)."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["markdown", "math_inline", "math_display"],
                        "description": (
                            "markdown: GitHub-flavored markdown. math_inline: TeX for $...$. "
                            "math_display: TeX for a $$...$$ block."
                        ),
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "For markdown, normal text. For math kinds, raw TeX only (no $ or $$). "
                            "Inside JSON strings, backslashes must be JSON-escaped (e.g. \\\\frac)."
                        ),
                    },
                },
                "required": ["kind", "body"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}

STRUCTURED_TOOL_SYSTEM_ADDENDUM = (
    "You must respond only by calling the tool deliver_chat_response. "
    "Do not put JSON or LaTeX-heavy content in plain assistant text.\n"
    "Split the answer into ordered segments: markdown for prose; for any mathematics, "
    "use math_inline or math_display segments with raw TeX in `body` (no surrounding $ or $$).\n"
    "In JSON, each backslash in TeX must be written as two backslashes."
)


def inject_structured_system_instruction(messages: List[dict]) -> List[dict]:
    """Append structured-output instructions to the (collapsed) system message."""
    out: List[dict] = []
    injected = False
    for m in messages:
        if (m.get("role") or "").lower() == "system":
            c = m.get("content")
            base = str(c) if c is not None else ""
            out.append(
                {
                    "role": "system",
                    "content": (base + "\n\n" + STRUCTURED_TOOL_SYSTEM_ADDENDUM).strip(),
                }
            )
            injected = True
        else:
            out.append(dict(m))
    if not injected:
        out.insert(0, {"role": "system", "content": STRUCTURED_TOOL_SYSTEM_ADDENDUM})
    return out


def normalize_segments(payload: object) -> List[Dict[str, str]]:
    if not isinstance(payload, dict):
        raise ValueError("tool payload must be a JSON object")
    segs = payload.get("segments")
    if not isinstance(segs, list):
        raise ValueError("tool payload missing segments array")
    out: List[Dict[str, str]] = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        kind = s.get("kind")
        if kind not in ("markdown", "math_inline", "math_display"):
            raise ValueError(f"invalid segment kind: {kind!r}")
        body = s.get("body")
        out.append({"kind": kind, "body": "" if body is None else str(body)})
    return out


def segments_to_markdown(segments: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for s in segments:
        kind = s["kind"]
        body = s["body"]
        if kind == "markdown":
            parts.append(body)
        elif kind == "math_inline":
            parts.append(f"${body}$")
        elif kind == "math_display":
            parts.append(f"\n\n$$\n{body.strip()}\n$$\n\n")
    return "".join(parts).strip()


def _openai_tool_definition() -> dict:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Submit the full assistant reply as ordered segments. Required for this turn."
            ),
            "parameters": DELIVER_CHAT_RESPONSE_SCHEMA,
        },
    }


def _extract_openai_tool_payload(data: dict) -> dict:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("OpenAI response has no choices")
    msg = choices[0].get("message") or {}
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict) or tc.get("type") != "function":
            continue
        fn = tc.get("function") or {}
        if fn.get("name") != TOOL_NAME:
            continue
        raw = fn.get("arguments")
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
    content = msg.get("content")
    raise ValueError(f"expected {TOOL_NAME} tool call; got content={content!r}")


def _extract_anthropic_tool_payload(data: dict) -> dict:
    for block in data.get("content") or []:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == TOOL_NAME
        ):
            inp = block.get("input")
            if isinstance(inp, dict):
                return inp
    raise ValueError(f"expected {TOOL_NAME} tool_use block in Anthropic response")


async def openai_structured_complete(
    model: str, messages: List[dict], api_key: str
) -> Tuple[str, List[Dict[str, str]]]:
    msgs = inject_structured_system_instruction(messages)
    body = {
        "model": model,
        "messages": msgs,
        "tools": [_openai_tool_definition()],
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        "parallel_tool_calls": False,
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=httpx.Timeout(30.0, read=300.0),
        )
        if r.status_code != 200:
            raise RuntimeError(r.text)
        data = r.json()
    payload = _extract_openai_tool_payload(data)
    segments = normalize_segments(payload)
    return segments_to_markdown(segments), segments


async def anthropic_structured_complete(
    model: str, messages: List[dict], api_key: str
) -> Tuple[str, List[Dict[str, str]]]:
    msgs = inject_structured_system_instruction(messages)
    system_text = ""
    api_messages: List[dict] = []
    for m in msgs:
        role = (m.get("role") or "").lower()
        if role == "system":
            c = m.get("content")
            chunk = str(c) if c is not None else ""
            system_text = (system_text + "\n\n" + chunk).strip() if system_text else chunk
        else:
            api_messages.append({"role": m["role"], "content": m["content"]})

    payload: dict = {
        "model": model,
        "max_tokens": 4096,
        "messages": api_messages,
        "tools": [
            {
                "name": TOOL_NAME,
                "description": (
                    "Submit the full assistant reply as ordered segments. Required for this turn."
                ),
                "input_schema": DELIVER_CHAT_RESPONSE_SCHEMA,
            }
        ],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
    }
    if system_text:
        payload["system"] = system_text

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=httpx.Timeout(30.0, read=300.0),
        )
        if r.status_code != 200:
            raise RuntimeError(r.text)
        data = r.json()
    tool_payload = _extract_anthropic_tool_payload(data)
    segments = normalize_segments(tool_payload)
    return segments_to_markdown(segments), segments
