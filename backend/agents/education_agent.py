"""LangChain education agent.

Given a topic + desired output format, this agent:
  * retrieves relevant context from the insertion ChromaDB collection (RAG),
  * optionally web-searches for up-to-date information,
  * drafts the educational material with an explicit chain-of-thought planner,
  * validates the output format (markdown / html / json / pdf),
  * writes the final artifact to ``data/generated/`` and returns the path.

Tool use is handled by ``create_tool_calling_agent`` so tools can be swapped
or extended without rewriting the orchestrator.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings

from . import metrics
from .config import (
    CHROMA_DIR,
    GENERATED_DIR,
    INSERTION_COLLECTION,
    education_agent_model,
    embedding_model,
)
from .helpers import validate_output, write_output_file
from .helpers.web_search import web_search_markdown


_PLANNER_PROMPT = """You are a senior instructional designer.
Produce a short plan (4-8 bullet points) for educational material on the
given topic. The plan must cover learning objectives, audience level,
key concepts, and a suggested structure. Do NOT write the final content yet."""


_SYSTEM_PROMPT = """You are Pluto's education-content agent.
Goal: produce high-quality educational content in the requested format
({format}). Always:

  1. Use the `rag_search` tool to retrieve grounded context from the local
     knowledge base. Call it at least once.
  2. If the topic may be time-sensitive or you are unsure, call `web_search`
     for supporting sources (max twice).
  3. Think step-by-step. Cite sources inline as [source: filename] or
     [source: web - title] when possible.
  4. When the draft is ready, call `validate_format` to check it parses in
     the requested format.
  5. Finally, call `finalize_output` with the validated content. Its return
     value contains the path to the saved file; include that path in your
     last message.

Only output plain content (no markdown fences around the whole thing) when
format is not markdown."""


def _strip_code_fence(text: str, fmt: str) -> str:
    if not isinstance(text, str):
        return text
    # Remove wrapping ```<lang>\n...\n``` fences that LLMs sometimes add.
    m = re.match(r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```\s*$", text.strip())
    if m:
        return m.group(1)
    return text


class EducationAgent:
    def __init__(self) -> None:
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._vector_store: Optional[Chroma] = None

    def _get_embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(model=embedding_model(), keep_alive=0)
        return self._embeddings

    def _get_vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=INSERTION_COLLECTION,
                embedding_function=self._get_embeddings(),
                persist_directory=CHROMA_DIR,
            )
        return self._vector_store

    def _plan(self, llm: ChatOllama, topic: str, audience: str) -> str:
        resp = llm.invoke(
            [
                SystemMessage(content=_PLANNER_PROMPT),
                HumanMessage(content=f"Topic: {topic}\nAudience: {audience}\nReturn the plan only."),
            ]
        )
        content = getattr(resp, "content", "") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return str(content).strip()

    def _build_agent(self, *, fmt: str, topic: str, plan: str, run_record: dict) -> AgentExecutor:
        vector_store = self._get_vector_store()

        @tool
        def rag_search(query: str, k: int = 4) -> str:
            """Retrieve ``k`` matching chunks from the local knowledge base.

            Returns a formatted string with each chunk's source and content.
            Always call this at least once to ground the answer.
            """
            try:
                docs = vector_store.similarity_search(query, k=max(1, min(int(k or 4), 10)))
            except Exception as e:
                metrics.log_event(run_record, "rag_error", error=str(e))
                return f"(rag_search failed: {e})"
            metrics.log_event(run_record, "rag_search", query=query, hits=len(docs))
            if not docs:
                return f"No KB results for: {query}"
            parts: List[str] = [f"KB results for: {query}"]
            for i, d in enumerate(docs, start=1):
                src = (d.metadata or {}).get("source", "unknown")
                title = (d.metadata or {}).get("title", "")
                parts.append(f"--- [{i}] source: {src}" + (f" | title: {title}" if title else ""))
                parts.append(d.page_content.strip())
            return "\n".join(parts)

        @tool
        def web_search(query: str, max_results: int = 5) -> str:
            """Run a DuckDuckGo web search and return title/url/snippet markdown."""
            metrics.log_event(run_record, "web_search", query=query)
            return web_search_markdown(query, max_results=max(1, min(int(max_results or 5), 8)))

        @tool
        def validate_format(content: str, format: str = fmt) -> dict:
            """Validate that ``content`` parses in ``format`` (markdown, html, json, pdf).

            Returns ``{ok, errors, sanitized}``. If ``ok`` is false, fix the
            content and call this tool again before finalizing.
            """
            result = validate_output(content, format or fmt)
            metrics.log_event(run_record, "validate_format", ok=result.ok, errors=result.errors)
            return {"ok": result.ok, "errors": result.errors, "format": result.format}

        @tool
        def finalize_output(content: str, format: str = fmt, base_name: Optional[str] = None) -> dict:
            """Persist the validated content to ``data/generated/``.

            Returns ``{path, format, chars}``. This must be the LAST tool call.
            """
            normalized = _strip_code_fence(content, format or fmt)
            validation = validate_output(normalized, format or fmt)
            if not validation.ok:
                metrics.log_event(run_record, "finalize_rejected", errors=validation.errors)
                return {
                    "ok": False,
                    "errors": validation.errors,
                    "message": "Content did not validate; fix and call finalize_output again.",
                }
            path = write_output_file(
                validation.sanitized or normalized,
                validation.format,
                out_dir=GENERATED_DIR,
                base_name=base_name or f"{topic[:60]}",
            )
            metrics.log_event(run_record, "finalized", path=path, format=validation.format)
            run_record["metrics"]["output_path"] = path
            run_record["metrics"]["output_format"] = validation.format
            run_record["metrics"]["output_chars"] = len(validation.sanitized or normalized)
            return {
                "ok": True,
                "path": path,
                "format": validation.format,
                "chars": len(validation.sanitized or normalized),
            }

        tools = [rag_search, web_search, validate_format, finalize_output]
        llm = ChatOllama(model=education_agent_model(), temperature=0.2, keep_alive=-1)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT.replace("{format}", fmt)),
                (
                    "human",
                    "Topic: {topic}\nAudience: {audience}\nRequested format: {format}\n\n"
                    "Instructional plan (chain-of-thought):\n{plan}\n\n"
                    "Now draft the content, use the tools to ground and validate it, and finalize.",
                ),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)

    def generate(
        self,
        *,
        topic: str,
        audience: str = "intermediate learners",
        fmt: str = "markdown",
        use_web: bool = True,
        extra_instructions: str = "",
    ) -> dict:
        fmt = (fmt or "markdown").strip().lower()
        if fmt in ("md",):
            fmt = "markdown"
        if fmt not in ("markdown", "html", "json", "pdf"):
            raise ValueError(f"Unsupported format: {fmt}. Use markdown, html, json, or pdf.")

        with metrics.track_run(
            "education",
            action="generate",
            metadata={
                "topic": topic,
                "audience": audience,
                "format": fmt,
                "use_web": use_web,
                "model": education_agent_model(),
                "embedding_model": embedding_model(),
            },
        ) as run:
            llm_plan = ChatOllama(model=education_agent_model(), temperature=0.2, keep_alive=-1)
            plan = self._plan(llm_plan, topic, audience)
            metrics.log_event(run, "plan_ready", chars=len(plan))

            executor = self._build_agent(fmt=fmt, topic=topic, plan=plan, run_record=run)
            try:
                output = executor.invoke(
                    {
                        "topic": topic + ("\n\nAdditional instructions: " + extra_instructions if extra_instructions else ""),
                        "audience": audience,
                        "format": fmt,
                        "plan": plan,
                    }
                )
            except Exception as e:
                run["status"] = "error"
                run["error"] = str(e)
                raise

            final_text = output.get("output", "") if isinstance(output, dict) else str(output)
            output_path = run["metrics"].get("output_path")

            # Fallback: if the agent forgot to call finalize_output, persist the final message directly.
            if not output_path and final_text.strip():
                metrics.log_event(run, "finalize_fallback")
                normalized = _strip_code_fence(final_text, fmt)
                validation = validate_output(normalized, fmt)
                if validation.ok:
                    output_path = write_output_file(
                        validation.sanitized or normalized,
                        validation.format,
                        out_dir=GENERATED_DIR,
                        base_name=topic[:60],
                    )
                    run["metrics"]["output_path"] = output_path
                    run["metrics"]["output_format"] = validation.format
                else:
                    run["metrics"]["validation_errors"] = validation.errors

            return {
                "topic": topic,
                "audience": audience,
                "format": fmt,
                "plan": plan,
                "agent_output": final_text,
                "output_path": output_path,
                "run_id": run["id"],
            }


_SINGLETON: Optional[EducationAgent] = None


def get_education_agent() -> EducationAgent:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = EducationAgent()
    return _SINGLETON
