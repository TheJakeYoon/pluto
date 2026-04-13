"""
Ollama local tool calling for structured JSON-shaped outputs.

See https://docs.ollama.com/capabilities/tool-calling — the model returns tool_calls;
arguments are already structured (dict), avoiding fragile free-form JSON in assistant text.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import ollama

_log = logging.getLogger("uvicorn.error")

from structured_chat import (
    DELIVER_CHAT_RESPONSE_SCHEMA,
    TOOL_NAME,
    inject_structured_system_instruction,
    normalize_segments,
    segments_to_markdown,
)


def _ollama_tool_spec(name: str, description: str, parameters: dict) -> dict:
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


def _coerce_arguments(raw: object) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"tool arguments must be object or JSON string, got {type(raw).__name__}")


async def ollama_structured_chat_complete(
    model: str,
    messages: List[dict],
    *,
    custom_tool: Optional[Dict[str, Any]],
    keep_alive: Union[float, str, int],
) -> Tuple[str, dict]:
    """
    Run a single non-streaming Ollama chat with exactly one tool.

    If custom_tool is None, uses deliver_chat_response (markdown/math segments) like cloud.
    If custom_tool is set (name, description, parameters), uses that JSON Schema instead.

    Returns (message.content string for UI/storage, structured dict for API clients).
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

    client = ollama.AsyncClient()
    t0 = time.perf_counter()
    resp = await client.chat(
        model=model,
        messages=msgs,
        tools=tools,
        stream=False,
        keep_alive=keep_alive,
    )
    wall_s = time.perf_counter() - t0
    info = resp.model_dump() if hasattr(resp, "model_dump") else {}
    td = info.get("total_duration")
    ed = info.get("eval_duration")
    ld = info.get("load_duration")
    ped = info.get("prompt_eval_duration")

    message = resp.message
    tcalls = message.tool_calls
    if tcalls:
        call = tcalls[0]
        fn = call.function
        tname = (fn.name or "").strip()
        args = _coerce_arguments(fn.arguments)

        if custom_tool:
            if tname != default_tool_name:
                raise ValueError(f"expected tool {default_tool_name!r}, got {tname!r}")
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

        if tname != TOOL_NAME:
            raise ValueError(f"expected tool {TOOL_NAME!r}, got {tname!r}")
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

    # No tool call: surface model text if present (easier debugging)
    content = (message.content or "").strip()
    if content:
        raise ValueError(
            "Ollama returned assistant text but no tool_calls; model or prompt may not support tools"
        )
    raise ValueError("Ollama returned no tool_calls and no content")
