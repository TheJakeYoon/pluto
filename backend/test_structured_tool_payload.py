"""Unit tests for structured tool-call payload extraction (no live LLM)."""

from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from structured_chat import (
    TOOL_NAME,
    _payload_from_structured_output_dict,
    coerce_default_deliver_tool_arguments,
    coerce_tool_arguments,
)


class CoerceToolArgumentsTests(unittest.TestCase):
    def test_valid_json_unchanged(self) -> None:
        s = '{"segments": [{"kind": "markdown", "body": "hello"}]}'
        d = coerce_tool_arguments(s)
        self.assertEqual(d["segments"][0]["body"], "hello")

    def test_repair_tab_times_latex(self) -> None:
        # Model writes \\times as \\t + imes inside a JSON string (invalid JSON without repair).
        bad = r'{"segments": [{"kind": "math_inline", "body": "\times 2"}]}'
        d = coerce_tool_arguments(bad)
        self.assertIn("body", d["segments"][0])
        self.assertIn("times", d["segments"][0]["body"])

    def test_latex_blob_cases_shorthand(self) -> None:
        # Model emits TeX in the tool-args channel instead of JSON (e.g. drops \\begin).
        blob = r"{cases} 2x+y=7 \\ x-y=2 \end{cases}"
        d = coerce_default_deliver_tool_arguments(blob)
        self.assertEqual(d["segments"][0]["kind"], "math_display")
        body = d["segments"][0]["body"]
        self.assertIn(r"\begin{cases}", body)
        self.assertIn(r"\end{cases}", body)

    def test_latex_blob_fallback_disabled_for_custom_tool(self) -> None:
        blob = r"{cases} 2"
        with self.assertRaises(ValueError):
            coerce_tool_arguments(blob, latex_blob_fallback=False)


class PayloadFromStructuredOutputTests(unittest.TestCase):
    def test_parsed_dict(self) -> None:
        payload = {"segments": [{"kind": "markdown", "body": "ok"}]}
        out = {"parsed": payload, "raw": None}
        got = _payload_from_structured_output_dict(out, tool_name=TOOL_NAME)
        self.assertEqual(got, payload)

    def test_raw_tool_calls_dict_args(self) -> None:
        raw = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": TOOL_NAME,
                    "args": {"segments": [{"kind": "markdown", "body": "from tool"}]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        out = {"parsed": None, "raw": raw, "parsing_error": None}
        got = _payload_from_structured_output_dict(out, tool_name=TOOL_NAME)
        self.assertEqual(got["segments"][0]["body"], "from tool")

    def test_raw_string_args_coerced(self) -> None:
        # LangChain validates AIMessage.args as dict; some providers still expose string args on raw dicts.
        class _RawWithStrArgs:
            content = ""
            tool_calls = [
                {
                    "name": TOOL_NAME,
                    "args": r'{"segments": [{"kind": "markdown", "body": "\\frac{a}{b}"}]}',
                }
            ]

        out = {"parsed": None, "raw": _RawWithStrArgs(), "parsing_error": None}
        got = _payload_from_structured_output_dict(
            out,
            tool_name=TOOL_NAME,
            coerce_string_arguments=coerce_tool_arguments,
        )
        self.assertEqual(got["segments"][0]["body"], r"\frac{a}{b}")

    def test_invalid_tool_calls_string_args_use_fallback(self) -> None:
        raw = AIMessage(
            content="",
            tool_calls=[],
            invalid_tool_calls=[
                {
                    "name": TOOL_NAME,
                    "args": r"{cases} a \\ b \end{cases}",
                    "id": "bad1",
                    "error": "JSONDecodeError",
                }
            ],
        )
        out = {"parsed": None, "raw": raw, "parsing_error": None}
        got = _payload_from_structured_output_dict(
            out,
            tool_name=TOOL_NAME,
            coerce_string_arguments=coerce_default_deliver_tool_arguments,
        )
        self.assertEqual(got["segments"][0]["kind"], "math_display")
        self.assertIn(r"\begin{cases}", got["segments"][0]["body"])


if __name__ == "__main__":
    unittest.main()
