"""
Ollama local structured output via LangChain ``ChatOllama.with_structured_output``.

- Models — structured output: https://docs.langchain.com/oss/python/langchain/models#structured-output
- Agents / ToolStrategy (same function-calling idea): https://docs.langchain.com/oss/python/langchain/structured-output
- LangChain tools: https://docs.langchain.com/oss/python/langchain/tools
- Ollama tool wire format: https://docs.ollama.com/capabilities/tool-calling

Uses ``method="function_calling"`` (forced tool). When LangChain's parser fails on tool
arguments, falls back to ``coerce_tool_arguments`` (JSON escape repair for LaTeX, etc.).

**LangChain quirk:** ``langchain_ollama`` parses tool-call argument strings with strict
``json.loads`` while building the assistant message. Malformed JSON never reaches our
coercers unless we patch ``_parse_arguments_from_tool_call`` (see
``_patch_langchain_ollama_tool_argument_parsing``).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama
import langchain_ollama.chat_models as _lc_ollama_chat

_log = logging.getLogger("uvicorn.error")

from structured_chat import (
    DELIVER_CHAT_RESPONSE_SCHEMA,
    TOOL_NAME,
    _dict_messages_to_langchain,
    _payload_from_structured_output_dict,
    coerce_default_deliver_tool_arguments,
    coerce_tool_arguments,
    inject_structured_system_instruction,
    normalize_segments,
    segments_to_markdown,
)


def _patch_langchain_ollama_tool_argument_parsing() -> None:
    """Let invalid tool JSON reach :func:`coerce_tool_arguments` instead of failing in LangChain."""
    if getattr(_lc_ollama_chat, "_pluto_tool_arg_parse_patched", False):
        return

    _orig = _lc_ollama_chat._parse_arguments_from_tool_call

    def _parse_arguments_from_tool_call(raw_tool_call: dict[str, Any]) -> Optional[dict[str, Any]]:
        try:
            return _orig(raw_tool_call)
        except OutputParserException:
            fn = str((raw_tool_call.get("function") or {}).get("name") or "").strip()
            arguments = (raw_tool_call.get("function") or {}).get("arguments")
            if not isinstance(arguments, str):
                raise
            try:
                if fn == TOOL_NAME:
                    return coerce_default_deliver_tool_arguments(arguments)
                return coerce_tool_arguments(arguments, latex_blob_fallback=False)
            except (TypeError, ValueError) as e:
                _log.warning(
                    "Ollama tool args still invalid after Pluto coercion (tool=%r): %s",
                    fn or "?",
                    e,
                )
                raise

    _lc_ollama_chat._parse_arguments_from_tool_call = _parse_arguments_from_tool_call
    _lc_ollama_chat._pluto_tool_arg_parse_patched = True


_patch_langchain_ollama_tool_argument_parsing()


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


def _raw_message_text(raw: object) -> str:
    if raw is None:
        return ""
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(block.get("text", block)) if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return (str(content) if content is not None else "").strip()


def _extract_json_text_from_assistant_content(text: str) -> str:
    """Extract JSON from raw assistant text, including ```json fenced blocks."""
    t = (text or "").strip()
    if not t:
        return ""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    # If prose leaked around JSON, take the broadest object/array span.
    starts = [idx for idx in (t.find("{"), t.find("[")) if idx >= 0]
    ends = [idx for idx in (t.rfind("}"), t.rfind("]")) if idx >= 0]
    if starts and ends and max(ends) > min(starts):
        return t[min(starts) : max(ends) + 1].strip()
    return t


def _array_wrapper_field_for_schema(parameters: dict) -> Optional[str]:
    props = parameters.get("properties") if isinstance(parameters, dict) else None
    if not isinstance(props, dict):
        return None
    for preferred in ("questions", "items", "batch", "practice_questions", "problems"):
        spec = props.get(preferred)
        if isinstance(spec, dict) and spec.get("type") == "array":
            return preferred
    array_fields = [
        name
        for name, spec in props.items()
        if isinstance(spec, dict) and spec.get("type") == "array"
    ]
    return array_fields[0] if len(array_fields) == 1 else None


def _coerce_plain_json_content_to_tool_args(raw: object, parameters: dict) -> Optional[dict]:
    """Fallback when a model returns JSON text instead of a forced tool call."""
    text = _raw_message_text(raw)
    json_text = _extract_json_text_from_assistant_content(text)
    if not json_text:
        return None
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            parsed = coerce_tool_arguments(json_text, latex_blob_fallback=False)
        except Exception:
            return None
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        field = _array_wrapper_field_for_schema(parameters)
        if field:
            return {field: parsed}
    return None


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
            coerce_string_arguments=(
                coerce_default_deliver_tool_arguments
                if not custom_tool
                else coerce_tool_arguments
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        if custom_tool and raw is not None:
            fallback_args = _coerce_plain_json_content_to_tool_args(raw, params)
            if fallback_args is not None:
                args = fallback_args
                _log.warning(
                    "Ollama structured_output recovered custom tool args from plain assistant JSON (model=%s tool=%s).",
                    model,
                    default_tool_name,
                )
            else:
                args = None
        else:
            args = None
        if args is None:
            snippet = ""
            if raw is not None:
                snippet_text = _raw_message_text(raw)
                if snippet_text:
                    snippet = snippet_text.replace("\n", " ")
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
