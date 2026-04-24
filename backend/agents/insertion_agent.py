"""LangChain insertion agent.

Pipeline for a single upload:
  1. The backend stores the file at ``data/uploads/<ts>-<name>``.
  2. The correct loader extracts text (and page images when needed).
  3. If the loader reports ``needs_vision``, a multimodal LLM transcribes the images.
  4. A LangChain ``AgentExecutor`` is given two tools:
        - ``summarize_document``: generates (title, summary, tags) metadata.
        - ``store_document``: chunks the text and upserts into Chroma.
     The agent's job is to call them in order so the document lands in the KB.
  5. The run is tracked by ``metrics.track_run`` so the UI can show status + stats.

Using an AgentExecutor (rather than raw chains) fulfils the
"use langchain agent to ensure this works properly" requirement and lets users
extend the agent with additional tools without rewriting the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from typing import List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import metrics
from .config import (
    CHROMA_DIR,
    INSERTION_COLLECTION,
    UPLOADS_DIR,
    embedding_model,
    insertion_agent_model,
    is_allowed_filename,
)
from .helpers import (
    DocumentLoadResult,
    load_file_to_text,
    vision_ocr_image,
    vision_ocr_pdf_pages,
)


_SYSTEM_PROMPT = """You are Pluto's document insertion agent.
Your job is to enrich the provided document excerpt with clean metadata
and then store it in the vector database via the available tools.

Always follow this plan:
  1. Call `summarize_document` ONCE to produce a short title, a 2-3 sentence
     summary, and 3-8 tags covering the content.
  2. Call `store_document` ONCE with that metadata so the text is chunked and
     embedded.
  3. After the store succeeds, reply with a short confirmation that includes
     the chunk count returned by `store_document`.

Never fabricate content that is not in the excerpt."""


class InsertionAgent:
    """Encapsulates the LangChain insertion agent and its helper state."""

    def __init__(self) -> None:
        self._vector_store: Optional[Chroma] = None
        self._embeddings: Optional[OllamaEmbeddings] = None

    # --- lazy singletons ---
    def _get_embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(model=embedding_model(), keep_alive=0)
        return self._embeddings

    def get_vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=INSERTION_COLLECTION,
                embedding_function=self._get_embeddings(),
                persist_directory=CHROMA_DIR,
            )
        return self._vector_store

    # --- file persistence ---
    @staticmethod
    def persist_upload(filename: str, data: bytes) -> str:
        if not is_allowed_filename(filename):
            raise ValueError(f"Rejected file type: {filename}")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        digest = hashlib.sha1(data).hexdigest()[:8]
        safe = os.path.basename(filename).replace(os.sep, "_")
        out_path = os.path.join(UPLOADS_DIR, f"{ts}-{digest}-{safe}")
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path

    # --- extraction (loader + optional vision) ---
    def extract_text(self, file_path: str, run_record: Optional[dict] = None) -> DocumentLoadResult:
        result = load_file_to_text(file_path)
        if run_record is not None:
            metrics.log_event(
                run_record,
                "loaded",
                loader=result.metadata.get("loader"),
                text_chars=len(result.text),
                needs_vision=result.needs_vision,
                pages=len(result.page_images),
                images=len(result.images),
            )
        if result.needs_vision:
            model = insertion_agent_model()
            if result.page_images:
                vision_text = vision_ocr_pdf_pages(result.page_images, model=model)
            elif result.images:
                parts = []
                for idx, png in enumerate(result.images, start=1):
                    parts.append(f"[Image {idx}]\n{vision_ocr_image(png, model=model)}")
                vision_text = "\n\n".join(parts)
            else:
                vision_text = ""
            if vision_text.strip():
                joiner = "\n\n" if result.text else ""
                result.text = f"{result.text}{joiner}{vision_text}".strip()
            if run_record is not None:
                metrics.log_event(run_record, "vision_ocr", chars=len(vision_text))
        return result

    # --- agent construction ---
    def _build_agent(self, file_path: str, extracted_text: str, loader_meta: dict) -> AgentExecutor:
        vector_store = self.get_vector_store()

        @tool
        def summarize_document(title: str, summary: str, tags: List[str]) -> dict:
            """Return a confirmation with the supplied title, summary, and tags.

            Use this ONCE to record the metadata you extracted from the excerpt.
            ``tags`` must be a JSON array of 3-8 short lowercase strings.
            """
            cleaned_tags = [t.strip().lower() for t in (tags or []) if isinstance(t, str) and t.strip()]
            return {
                "title": (title or "").strip() or os.path.basename(file_path),
                "summary": (summary or "").strip(),
                "tags": cleaned_tags[:8],
            }

        @tool
        def store_document(title: str, summary: str, tags: List[str]) -> dict:
            """Chunk and embed the loaded document into ChromaDB.

            The text comes from the pre-loaded file; the agent only needs to
            supply the metadata (``title``, ``summary``, ``tags``).
            Returns the number of chunks stored.
            """
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_text(extracted_text)
            base_meta = {
                "source": os.path.basename(file_path),
                "source_path": file_path,
                "title": (title or "").strip() or os.path.basename(file_path),
                "summary": (summary or "").strip(),
                "tags": ",".join([t.strip().lower() for t in (tags or []) if isinstance(t, str) and t.strip()][:8]),
                "loader": loader_meta.get("loader", ""),
                "stored_at": datetime.now().astimezone().isoformat(),
            }
            docs = [Document(page_content=c, metadata=dict(base_meta)) for c in chunks if c.strip()]
            if not docs:
                return {"stored": 0, "message": "No non-empty chunks to store."}
            vector_store.add_documents(docs)
            return {"stored": len(docs), "collection": INSERTION_COLLECTION}

        tools = [summarize_document, store_document]
        llm = ChatOllama(model=insertion_agent_model(), temperature=0, keep_alive=-1)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", "File: {filename}\nLoader: {loader}\n\nExcerpt (truncated):\n{excerpt}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)

    # --- public entrypoint ---
    def ingest_file(self, file_path: str, *, force_vision: bool = False) -> dict:
        """Run extraction + agent over a single file. Returns a result dict."""
        filename = os.path.basename(file_path)
        with metrics.track_run(
            "insertion",
            action="ingest",
            metadata={"filename": filename, "path": file_path, "model": insertion_agent_model()},
        ) as run:
            load_result = self.extract_text(file_path, run_record=run)
            text = load_result.text.strip()
            if not text:
                run["status"] = "error"
                run["error"] = "No text extracted from file."
                raise RuntimeError("No text extracted from file.")

            excerpt = text[:4000]
            executor = self._build_agent(file_path, text, load_result.metadata)
            try:
                output = executor.invoke(
                    {
                        "filename": filename,
                        "loader": load_result.metadata.get("loader", "unknown"),
                        "excerpt": excerpt,
                    }
                )
                agent_output = output.get("output", "") if isinstance(output, dict) else str(output)
            except Exception as e:
                # Fallback: still store the text directly so the upload is not lost.
                metrics.log_event(run, "agent_fallback", error=str(e))
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_text(text)
                docs = [
                    Document(
                        page_content=c,
                        metadata={
                            "source": filename,
                            "source_path": file_path,
                            "loader": load_result.metadata.get("loader", ""),
                            "fallback": True,
                        },
                    )
                    for c in chunks
                    if c.strip()
                ]
                self.get_vector_store().add_documents(docs)
                agent_output = f"Agent error ({e}); stored {len(docs)} chunks directly."
                run["metrics"]["stored_chunks"] = len(docs)
                run["metrics"]["agent_fallback"] = True
                return {
                    "file": filename,
                    "path": file_path,
                    "stored_chunks": len(docs),
                    "agent_output": agent_output,
                    "fallback": True,
                    "warnings": load_result.warnings,
                    "run_id": run["id"],
                }

            # Count what was stored by inspecting steps
            stored_chunks = 0
            intermediate = output.get("intermediate_steps") if isinstance(output, dict) else None
            if intermediate:
                for _, obs in intermediate:
                    if isinstance(obs, dict) and "stored" in obs:
                        stored_chunks = max(stored_chunks, int(obs.get("stored") or 0))
            run["metrics"]["stored_chunks"] = stored_chunks
            run["metrics"]["chars"] = len(text)
            return {
                "file": filename,
                "path": file_path,
                "stored_chunks": stored_chunks,
                "agent_output": agent_output,
                "fallback": False,
                "warnings": load_result.warnings,
                "run_id": run["id"],
            }


_SINGLETON: Optional[InsertionAgent] = None


def get_insertion_agent() -> InsertionAgent:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = InsertionAgent()
    return _SINGLETON
