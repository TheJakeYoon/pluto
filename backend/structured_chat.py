"""
Structured assistant replies via LangChain structured output on chat models.

Uses ``ChatOpenAI.with_structured_output`` / ``ChatOllama.with_structured_output``
(``method="function_calling"``) — the standalone-model API documented at
https://docs.langchain.com/oss/python/langchain/models#structured-output
(same underlying strategy as agent ``ToolStrategy`` in
https://docs.langchain.com/oss/python/langchain/structured-output).

Cloud APIs parse tool/function arguments as JSON, which avoids asking the model to
emit raw JSON in assistant text (a common source of invalid escapes and LaTeX
backslash corruption). Math is isolated in typed segments so TeX stays in
dedicated JSON string values with normal JSON escaping.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

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
    "In JSON string values, every backslash must be doubled (e.g. `\\\\frac` for \\frac). "
    "A single `\\t` inside a string is a TAB character — so write `\\\\times` for \\times, not `\\times`."
)


def _repair_invalid_json_escape_sequences(s: str) -> str:
    """
    Fix common LLM mistake: LaTeX ``\\times`` written as ``\\t`` + ``imes`` in JSON strings.

    Inside double-quoted JSON strings only: if ``\\`` is not followed by a valid JSON
    escape introducer, emit an extra ``\\`` so the following character is literal.
    """
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


def _looks_like_non_json_object_brace(t: str) -> bool:
    """True when ``t`` starts like ``{foo`` (not ``{"key"``) — common LaTeX / typo, not JSON."""
    if not t.startswith("{"):
        return False
    i = 1
    n = len(t)
    while i < n and t[i].isspace():
        i += 1
    return i >= n or t[i] != '"'


def _latex_blob_tool_string_heuristic(s: str) -> bool:
    """Heuristic: string is probably math/LaTeX leaked into tool args instead of JSON."""
    t = s.strip()
    if _looks_like_non_json_object_brace(t):
        return True
    if re.search(r"\\end\s*\{\s*cases\s*\}|\\begin\s*\{\s*cases\s*\}|\{\s*cases\s*\}", t):
        return True
    return False


def _recover_cases_shorthand(body: str) -> str:
    """Turn ``{cases} ... \\end{cases}`` (missing ``\\begin``) into valid amsmath cases body."""
    t = body.strip()
    m = re.match(r"^\{\s*cases\s*\}", t, re.IGNORECASE)
    if m:
        t = "\\begin{cases}" + t[m.end() :]
    return t


def _fallback_segments_payload_from_latex_blob(s: str) -> dict:
    """Last resort for ``deliver_chat_response`` when the model emits TeX, not JSON."""
    body = _recover_cases_shorthand(s.strip())
    return {"segments": [{"kind": "math_display", "body": body}]}


def coerce_tool_arguments(raw: object, *, latex_blob_fallback: bool = False) -> dict:
    """Normalize LangChain / API tool ``arguments`` to a dict (robust to bad JSON escapes).

    When ``latex_blob_fallback`` is True (default ``deliver_chat_response`` tool only), a string
    that is still not JSON but looks like LaTeX (e.g. ``{cases} ... \\end{cases}``) is wrapped as
    a single ``math_display`` segment so the reply can render instead of failing JSON parse.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        raise TypeError(f"tool arguments must be object or JSON string, got {type(raw).__name__}")
    s = raw.strip()
    if not s:
        return {}
    fixed = _repair_invalid_json_escape_sequences(s)
    try:
        out = json.loads(fixed)
    except json.JSONDecodeError:
        try:
            out = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            try:
                out = ast.literal_eval(fixed)
            except (SyntaxError, ValueError) as e:
                if latex_blob_fallback and _latex_blob_tool_string_heuristic(s):
                    return _fallback_segments_payload_from_latex_blob(s)
                raise ValueError(f"tool arguments are not valid JSON after escape repair: {e}") from e
    if not isinstance(out, dict):
        raise ValueError("tool arguments JSON must decode to an object")
    return out


def coerce_default_deliver_tool_arguments(raw: object) -> dict:
    """Like :func:`coerce_tool_arguments` with ``latex_blob_fallback=True`` for ``deliver_chat_response``."""
    return coerce_tool_arguments(raw, latex_blob_fallback=True)


def _dict_messages_to_langchain(messages: List[dict]) -> list:
    """API-style role/content dicts → LangChain messages (Ollama / OpenAI chat)."""
    out: list = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        text = str(m.get("content") if m.get("content") is not None else "")
        if role == "system":
            out.append(SystemMessage(content=text))
        elif role == "assistant":
            out.append(AIMessage(content=text))
        elif role == "tool":
            out.append(
                ToolMessage(
                    content=text,
                    tool_call_id=str(m.get("tool_call_id") or m.get("id") or ""),
                )
            )
        else:
            out.append(HumanMessage(content=text))
    return out


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


def _payload_from_structured_output_dict(
    out: dict,
    *,
    tool_name: str,
    coerce_string_arguments: Optional[Callable[[Any], dict]] = None,
) -> dict:
    """Read parsed tool arguments from ``with_structured_output(..., include_raw=True)`` result."""
    parsed = out.get("parsed")
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            data = parsed.model_dump()
        elif isinstance(parsed, dict):
            data = parsed
        else:
            data = dict(parsed)
        if isinstance(data, dict):
            return data
    raw = out.get("raw")
    pe = out.get("parsing_error")
    if raw is None:
        raise ValueError(
            f"expected structured tool {tool_name!r}; no parsed payload and no raw message "
            f"(parsing_error={pe!r})"
        )

    def _all_tool_call_dicts() -> list:
        merged: list = []
        for tc in getattr(raw, "tool_calls", None) or []:
            merged.append(dict(tc) if not isinstance(tc, dict) else tc)
        for itc in getattr(raw, "invalid_tool_calls", None) or []:
            merged.append(dict(itc) if not isinstance(itc, dict) else itc)
        return merged

    tcalls = _all_tool_call_dicts()
    for tc in tcalls:
        if (tc.get("name") or "").strip() != tool_name:
            continue
        ra = tc.get("args")
        if isinstance(ra, dict):
            return ra
        if coerce_string_arguments is not None:
            return coerce_string_arguments(ra)
        if isinstance(ra, str):
            return coerce_tool_arguments(ra, latex_blob_fallback=False)
        raise ValueError(f"tool arguments must be object or string, got {type(ra).__name__}")
    c = getattr(raw, "content", None)
    raise ValueError(f"expected {tool_name!r} tool call; got content={c!r} parsing_error={pe!r}")


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


def _openai_kwargs_structured_with_optional_reasoning(
    *,
    reasoning_effort: Optional[str],
) -> dict:
    """OpenAI rejects ``reasoning_effort`` + function tools on ``/v1/chat/completions`` for some models (e.g. gpt-5.4-nano).

    LangChain must use the Responses API (``use_responses_api=True``, ``output_version='responses/v1'``) so tool calls
    and reasoning share one supported path.
    """
    if not reasoning_effort:
        return {}
    return {
        "reasoning_effort": reasoning_effort,
        "use_responses_api": True,
        "output_version": "responses/v1",
    }


async def openai_structured_complete(
    model: str,
    messages: List[dict],
    api_key: str,
    *,
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    msgs = inject_structured_system_instruction(messages)
    lc = _dict_messages_to_langchain(msgs)
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        timeout=httpx.Timeout(30.0, read=300.0),
        max_retries=2,
        **_openai_kwargs_structured_with_optional_reasoning(reasoning_effort=reasoning_effort),
    )
    structured = llm.with_structured_output(
        _openai_tool_definition(),
        method="function_calling",
        include_raw=True,
    )
    try:
        out = await structured.ainvoke(lc)
    except Exception as e:
        raise RuntimeError(str(e)) from e
    payload = _payload_from_structured_output_dict(
        out,
        tool_name=TOOL_NAME,
        coerce_string_arguments=coerce_default_deliver_tool_arguments,
    )
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


def _patch_langchain_core_parse_tool_call() -> None:
    """When OpenAI-format ``function.arguments`` is not valid JSON, apply Pluto coercion (same as Ollama patch)."""
    from langchain_core.exceptions import OutputParserException
    from langchain_core.messages.tool import tool_call as create_tool_call
    from langchain_core.output_parsers import openai_tools as ot

    if getattr(ot, "_pluto_parse_tool_call_patched", False):
        return

    _orig = ot.parse_tool_call

    def parse_tool_call(
        raw_tool_call: dict[str, Any],
        *,
        partial: bool = False,
        strict: bool = False,
        return_id: bool = True,
    ) -> Any:
        try:
            return _orig(
                raw_tool_call,
                partial=partial,
                strict=strict,
                return_id=return_id,
            )
        except OutputParserException:
            if partial:
                raise
            fn = str((raw_tool_call.get("function") or {}).get("name") or "").strip()
            argstr = (raw_tool_call.get("function") or {}).get("arguments")
            if not isinstance(argstr, str):
                raise
            if fn == TOOL_NAME:
                coerced = coerce_default_deliver_tool_arguments(argstr)
            else:
                coerced = coerce_tool_arguments(argstr, latex_blob_fallback=False)
            parsed: dict[str, Any] = {
                "name": raw_tool_call["function"]["name"] or "",
                "args": coerced or {},
            }
            if return_id:
                parsed["id"] = raw_tool_call.get("id")
                return create_tool_call(**parsed)
            return parsed

    ot.parse_tool_call = parse_tool_call
    ot._pluto_parse_tool_call_patched = True


_patch_langchain_core_parse_tool_call()
