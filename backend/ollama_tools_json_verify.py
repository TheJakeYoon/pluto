"""
Verify local Ollama + LangChain ChatOllama: tool calling and JSON-schema output (LaTeX in strings).

Uses the same stack as main.py (langchain-ollama). Run from repo:

  cd backend && ./venv/bin/python ollama_tools_json_verify.py
  OLLAMA_TEST_MODEL=gemma4:26b ./venv/bin/python ollama_tools_json_verify.py

For Gemma 4 + structured JSON, Ollama expects think=false alongside format (see ollama/ollama#15260);
ChatOllama maps ``reasoning=False`` to ``think=false``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

_SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

LATEX_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "formula_latex": {
            "type": "string",
            "description": "LaTeX snippet; backslashes must be JSON-escaped",
        },
    },
    "required": ["title", "formula_latex"],
}


class Multiply(BaseModel):
    """Tool schema for a trivial multiply; models emit tool_calls with JSON args."""

    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")


def _default_model_from_settings() -> str:
    try:
        with open(_SETTINGS, encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("default_chat_model")
        if isinstance(v, str) and v.strip():
            return v.strip()
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return "gpt-oss:20b"


def _is_gemma4(model: str) -> bool:
    return "gemma4" in (model or "").lower()


def _chat_kwargs_for_model(model: str) -> dict[str, Any]:
    """Match main.py Gemma 4 sampling when verifying that model family."""
    if _is_gemma4(model):
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    return {"temperature": 0}


def _think_off_for_gemma4_json(model: str) -> dict[str, Any]:
    if _is_gemma4(model):
        return {"reasoning": False}
    return {}


def _parse_json_content(raw: str) -> dict[str, Any]:
    """Parse model JSON; tolerate a single fenced markdown block."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return json.loads(s)


async def verify_tool_calling(model: str) -> dict[str, Any]:
    llm = ChatOllama(
        model=model,
        num_predict=256,
        keep_alive="5m",
        **_chat_kwargs_for_model(model),
    )
    bound = llm.bind_tools([Multiply])
    msg = HumanMessage(
        content=(
            "You must use the Multiply tool and nothing else for the arithmetic. "
            "What is 12 times 34?"
        )
    )
    ai = await bound.ainvoke([msg])
    calls = getattr(ai, "tool_calls", None) or []
    if not calls:
        return {
            "ok": False,
            "error": "No tool_calls on assistant message",
            "content_preview": (ai.content or "")[:400],
        }
    tc = calls[0]
    name = tc.get("name")
    args = tc.get("args") or {}
    if name != "Multiply":
        return {"ok": False, "error": f"Unexpected tool name {name!r}", "tool_calls": calls}
    try:
        a, b = int(args["a"]), int(args["b"])
    except (KeyError, TypeError, ValueError) as e:
        return {"ok": False, "error": f"Bad tool args: {e}", "tool_calls": calls}
    if a * b != 12 * 34:
        return {"ok": False, "error": f"Wrong product {a}*{b}", "tool_calls": calls}
    return {"ok": True, "tool": "Multiply", "args": {"a": a, "b": b}}


async def verify_json_latex(model: str) -> dict[str, Any]:
    llm = ChatOllama(
        model=model,
        format=LATEX_JSON_SCHEMA,
        num_predict=512,
        keep_alive="5m",
        **_chat_kwargs_for_model(model),
        **_think_off_for_gemma4_json(model),
    )
    msg = HumanMessage(
        content=(
            "Reply with JSON only (no markdown) matching the schema. "
            'title: short label. formula_latex: the quadratic formula using LaTeX, '
            r"e.g. x = \frac{-b \pm \sqrt{b^2-4ac}}{2a} — in JSON strings every \\ must be doubled."
        )
    )
    ai = await llm.ainvoke([msg])
    raw = ai.content if isinstance(ai.content, str) else str(ai.content)
    try:
        data = _parse_json_content(raw)
    except json.JSONDecodeError as e:
        return {
            "ok": False,
            "error": f"json.loads failed: {e}",
            "raw_preview": raw[:800],
        }
    title = data.get("title")
    latex = data.get("formula_latex")
    if not isinstance(title, str) or not isinstance(latex, str):
        return {"ok": False, "error": "title/formula_latex not strings", "parsed": data}
    # Valid JSON already decoded escapes once — expect recognizable math notation in the string.
    if not re.search(r"frac|sqrt|\\\\|±|\\pm|b\^2|b²", latex, re.IGNORECASE):
        return {"ok": False, "error": "formula_latex missing expected TeX or quadratic markers", "parsed": data}
    return {"ok": True, "parsed": data}


async def run_ollama_tools_json_verification(model: str) -> dict[str, Any]:
    tools = await verify_tool_calling(model)
    jsn = await verify_json_latex(model)
    return {
        "model": model,
        "tool_calling": tools,
        "json_latex": jsn,
        "all_ok": bool(tools.get("ok") and jsn.get("ok")),
    }


async def _async_main(model: str) -> int:
    try:
        out = await run_ollama_tools_json_verification(model)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0 if out.get("all_ok") else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "model",
        nargs="?",
        default=os.environ.get("OLLAMA_TEST_MODEL") or _default_model_from_settings(),
        help="Ollama model id (default: OLLAMA_TEST_MODEL or settings.json default_chat_model)",
    )
    args = p.parse_args()
    raise SystemExit(asyncio.run(_async_main(args.model)))


if __name__ == "__main__":
    main()
