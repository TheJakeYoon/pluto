"""
Ollama local structured output via LangChain ``ChatOllama.with_structured_output``.

- Models — structured output: https://docs.langchain.com/oss/python/langchain/models#structured-output
- Agents / ToolStrategy (same function-calling idea): https://docs.langchain.com/oss/python/langchain/structured-output
- LangChain tools: https://docs.langchain.com/oss/python/langchain/tools
- Ollama tool wire format: https://docs.ollama.com/capabilities/tool-calling

Uses ``method="function_calling"`` (forced tool). When LangChain's parser fails on tool
arguments, falls back to ``_coerce_arguments`` (JSON escape repair for LaTeX, etc.).
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_ollama import ChatOllama

_log = logging.getLogger("uvicorn.error")

from structured_chat import (
    DELIVER_CHAT_RESPONSE_SCHEMA,
    TOOL_NAME,
    _dict_messages_to_langchain,
    _payload_from_structured_output_dict,
    inject_structured_system_instruction,
    normalize_segments,
    segments_to_markdown,
)


def _is_gemma4_model(model: str) -> bool:
    return "gemma4" in (model or "").lower()


def _ollama_sampling_kwargs(model: str) -> dict[str, Any]:
    if _is_gemma4_model(model):
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    return {"temperature": 0}


def _ollama_tool_spec(name: str, description: str, parameters: dict) -> dict:
    """OpenAI-style function tool dict; LangChain ``bind_tools`` accepts this shape."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Structured response via {name}.",
            "parameters": parameters,
        },
    }


def _inject_custom_tool_instruction(messages: List[dict], tool_name: str, tool_description: str) -> List[dict]:
    addendum = (
        f"You must respond only by calling the tool `{tool_name}`. "
        "Do not put the main answer in plain assistant text; use the tool arguments only.\n"
        "String values may contain LaTeX: every backslash must be JSON-valid — e.g. write `\\\\frac` for \\frac. "
        "A single `\\t` before `imes` is read as a TAB character in JSON and will break parsing; use `\\\\times` for \\times.\n"
    )
    if (tool_description or "").strip():
        addendum += f"Tool: {tool_description.strip()}\n"
    out: List[dict] = []
    injected = False
    for m in messages:
        if (m.get("role") or "").lower() == "system":
            c = m.get("content")
            base = str(c) if c is not None else ""
            out.append({"role": "system", "content": (base + "\n\n" + addendum).strip()})
            injected = True
        else:
            out.append(dict(m))
    if not injected:
        out.insert(0, {"role": "system", "content": addendum.strip()})
    return out


def _repair_invalid_json_escape_sequences(s: str) -> str:
    """
    Fix common LLM mistake: LaTeX ``\\times`` written as ``\\times`` in prose but in JSON
    appears as ``\\t`` + ``imes`` which JSON parses as TAB + ``imes`` (invalid).

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

        # Inside JSON string
        if ch == '"' and _backslashes_before(i) % 2 == 0:
            in_str = False
            out.append(ch)
            i += 1
            continue

        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            # JSON allows ``\\t`` as TAB — but models often write LaTeX ``\\times`` as ``\\t``+``imes``.
            # If ``t`` is followed by a letter, treat as LaTeX (not a tab) and duplicate the backslash.
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
            # Invalid JSON escape (e.g. LaTeX \alpha, \times): duplicate backslash
            out.append("\\")
            out.append("\\")
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _coerce_arguments(raw: object) -> dict:
    """Normalize Ollama / LangChain tool ``arguments`` to a dict (robust to bad JSON)."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        raise TypeError(f"tool arguments must be object or JSON string, got {type(raw).__name__}")
    s = raw.strip()
    if not s:
        return {}
    # Always run escape repair before json.loads: valid JSON is unchanged, but ``\\times`` written
    # as ``\\t``+``imes`` would otherwise decode as TAB without error.
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
                raise ValueError(f"tool arguments are not valid JSON after escape repair: {e}") from e
    if not isinstance(out, dict):
        raise ValueError("tool arguments JSON must decode to an object")
    return out


async def ollama_structured_chat_complete(
    model: str,
    messages: List[dict],
    *,
    custom_tool: Optional[Dict[str, Any]],
    keep_alive: Union[float, str, int],
    thinking: bool = True,
) -> Tuple[str, dict]:
    """
    Single non-streaming completion: LangChain ``ChatOllama.with_structured_output``
    (``method="function_calling"``).

    Default tool: ``deliver_chat_response`` (segments → markdown). Custom: ``structured_tool`` schema.
    """
    if custom_tool:
        name = str(custom_tool.get("name") or "").strip()
        if not name:
            raise ValueError("structured_tool.name is required")
        params = custom_tool.get("parameters")
        if not isinstance(params, dict):
            raise ValueError("structured_tool.parameters must be a JSON Schema object")
        desc = str(custom_tool.get("description") or "")
        tools = [_ollama_tool_spec(name, desc, params)]
        msgs = _inject_custom_tool_instruction(messages, name, desc)
        default_tool_name = name
    else:
        tools = [
            _ollama_tool_spec(
                TOOL_NAME,
                "Submit the full assistant reply as ordered segments.",
                DELIVER_CHAT_RESPONSE_SCHEMA,
            )
        ]
        msgs = inject_structured_system_instruction(messages)
        default_tool_name = TOOL_NAME

    lc_messages = _dict_messages_to_langchain(msgs)
    llm = ChatOllama(
        model=model,
        keep_alive=keep_alive,
        **_ollama_sampling_kwargs(model),
    )
    schema = tools[0]
    structured = llm.with_structured_output(
        schema,
        method="function_calling",
        include_raw=True,
    )

    invoke_kw: dict[str, Any] = {}
    if _is_gemma4_model(model):
        # With thinking off, force think=false for reliable tool JSON on Gemma 4.
        if not thinking:
            invoke_kw["reasoning"] = False
    else:
        invoke_kw["reasoning"] = bool(thinking)

    t0 = time.perf_counter()
    try:
        out = await structured.ainvoke(lc_messages, **invoke_kw)
    except Exception as e:
        raise ValueError(f"Ollama structured_output invoke failed (model={model!r}): {e}") from e
    wall_s = time.perf_counter() - t0
    raw = out.get("raw")
    rm = getattr(raw, "response_metadata", None) or {} if raw is not None else {}
    td = rm.get("total_duration")
    ed = rm.get("eval_duration")
    ld = rm.get("load_duration")
    ped = rm.get("prompt_eval_duration")

    try:
        args = _payload_from_structured_output_dict(
            out,
            tool_name=default_tool_name,
            coerce_string_arguments=_coerce_arguments,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        snippet = ""
        if raw is not None:
            raw_c = getattr(raw, "content", None)
            if isinstance(raw_c, str):
                snippet = raw_c.strip()
            elif isinstance(raw_c, list):
                snippet = "".join(
                    str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in raw_c
                ).strip()
            else:
                snippet = (str(raw_c) if raw_c is not None else "").strip()
            if snippet:
                snippet = snippet.replace("\n", " ")
                if len(snippet) > 420:
                    snippet = snippet[:420] + " …"
                snippet = f" Assistant text (excerpt): {snippet!r}."
        raise ValueError(
            f"Could not parse tool arguments for {default_tool_name!r} (model={model!r}): {e}. "
            "If values contain LaTeX, ensure JSON strings use doubled backslashes (e.g. \\\\frac)."
            f"{snippet}"
        ) from e

    if raw is not None:
        raw_names = {
            (tc.get("name") or "").strip()
            for tc in getattr(raw, "tool_calls", None) or []
            if (tc.get("name") or "").strip()
        }
        if raw_names and default_tool_name not in raw_names:
            raise ValueError(
                f"expected tool {default_tool_name!r}, got {sorted(raw_names)!r} (model={model!r})"
            )
    tname = default_tool_name

    if custom_tool:
        text = json.dumps(args, ensure_ascii=False)
        _log.info(
            "Ollama structured_output finished: model=%s tool=%s (custom JSON schema) wall_s=%.3f "
            "ollama_total_ms=%.2f ollama_eval_ms=%.2f ollama_load_ms=%.2f ollama_prompt_eval_ms=%.2f",
            model,
            tname,
            wall_s,
            (td or 0) / 1e6,
            (ed or 0) / 1e6,
            (ld or 0) / 1e6,
            (ped or 0) / 1e6,
        )
        return text, {"tool_name": tname, "arguments": args}

    segments = normalize_segments(args)
    md = segments_to_markdown(segments)
    _log.info(
        "Ollama structured_output finished: model=%s tool=%s (JSON segments to markdown) wall_s=%.3f "
        "ollama_total_ms=%.2f ollama_eval_ms=%.2f ollama_load_ms=%.2f ollama_prompt_eval_ms=%.2f",
        model,
        tname,
        wall_s,
        (td or 0) / 1e6,
        (ed or 0) / 1e6,
        (ld or 0) / 1e6,
        (ped or 0) / 1e6,
    )
    return md, {"segments": segments}
