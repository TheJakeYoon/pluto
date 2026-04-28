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
import logging
import os
import re
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

logger = logging.getLogger("uvicorn.error")


def _metadata_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _infer_subject(title: str, tags: List[str], text: str) -> str:
    haystack = " ".join([title or "", " ".join(tags or []), text[:3000]]).lower()
    subject_keywords = [
        ("precalculus", "precalculus"),
        ("calculus", "calculus"),
        ("statistics", "statistics"),
        ("computer science", "computer science"),
        ("biology", "biology"),
        ("chemistry", "chemistry"),
        ("physics", "physics"),
        ("english", "english"),
        ("history", "history"),
        ("economics", "economics"),
        ("psychology", "psychology"),
    ]
    for needle, subject in subject_keywords:
        if needle in haystack:
            return subject
    return tags[0] if tags else ""


def _section_for_chunk(chunk: str) -> str:
    for line in chunk.splitlines():
        cleaned = line.strip(" \t#:-")
        if not cleaned:
            continue
        if len(cleaned) <= 90 and (
            cleaned.isupper()
            or re.match(r"^(chapter|unit|section|part|topic|course framework|sample|appendix)\b", cleaned, re.I)
        ):
            return cleaned[:120]
    return ""


def _page_range_for_chunk(chunk: str) -> str:
    pages = [int(x) for x in re.findall(r"\[Page\s+(\d+)\]", chunk, flags=re.I)]
    if not pages:
        return ""
    lo, hi = min(pages), max(pages)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _pages_inserted_from_metadata(meta: dict) -> Optional[int]:
    value = meta.get("ocr_pages_processed") or meta.get("page_count")
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


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
        logger.info(
            "Insertion upload persisted original=%s path=%s bytes=%d sha1=%s",
            filename,
            out_path,
            len(data),
            digest,
        )
        return out_path

    # --- extraction (loader + optional vision) ---
    def extract_text(self, file_path: str, run_record: Optional[dict] = None) -> DocumentLoadResult:
        extract_start = time.perf_counter()
        logger.info("Insertion extract start file=%s path=%s", os.path.basename(file_path), file_path)
        def _progress(event: dict) -> None:
            if run_record is None:
                return
            if event.get("event") == "pdf_ocr_page":
                pages_done = int(event.get("pages_inserted") or event.get("page") or 0)
                page_count = int(event.get("page_count") or 0)
                run_record["metrics"]["pages_inserted"] = pages_done
                run_record["metrics"]["page_count"] = page_count
                run_record["metrics"]["ocr_chars"] = int(event.get("cumulative_chars") or 0)
                metrics.log_event(
                    run_record,
                    "pdf_ocr_page",
                    page=event.get("page"),
                    page_count=page_count,
                    pages_inserted=pages_done,
                    chars=event.get("chars"),
                    cumulative_chars=event.get("cumulative_chars"),
                )

        result = load_file_to_text(file_path, progress_callback=_progress)
        logger.info(
            "Insertion extract loaded file=%s loader=%s text_chars=%d warnings=%d needs_vision=%s pages=%d images=%d elapsed_s=%.2f",
            os.path.basename(file_path),
            result.metadata.get("loader"),
            len(result.text),
            len(result.warnings),
            result.needs_vision,
            len(result.page_images),
            len(result.images),
            time.perf_counter() - extract_start,
        )
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
            vision_start = time.perf_counter()
            logger.info(
                "Insertion vision fallback start file=%s model=%s page_images=%d images=%d",
                os.path.basename(file_path),
                model,
                len(result.page_images),
                len(result.images),
            )
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
            logger.info(
                "Insertion vision fallback complete file=%s chars=%d total_text_chars=%d elapsed_s=%.2f",
                os.path.basename(file_path),
                len(vision_text),
                len(result.text),
                time.perf_counter() - vision_start,
            )
        logger.info(
            "Insertion extract complete file=%s final_text_chars=%d elapsed_s=%.2f",
            os.path.basename(file_path),
            len(result.text),
            time.perf_counter() - extract_start,
        )
        return result

    # --- agent construction ---
    def _build_agent(
        self,
        file_path: str,
        extracted_text: str,
        loader_meta: dict,
        *,
        source_url: Optional[str] = None,
        run_record: Optional[dict] = None,
    ) -> AgentExecutor:
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
            clean_tags = [t.strip().lower() for t in (tags or []) if isinstance(t, str) and t.strip()][:8]
            clean_title = (title or "").strip() or os.path.basename(file_path)
            clean_summary = (summary or "").strip()
            subject = _infer_subject(clean_title, clean_tags, extracted_text)
            base_meta = {
                "source": os.path.basename(file_path),
                "source_path": file_path,
                "source_url": source_url or "",
                "title": clean_title,
                "summary": clean_summary,
                "subject": subject,
                "tags": ",".join(clean_tags),
                "loader": loader_meta.get("loader", ""),
                "document_type": os.path.splitext(file_path)[1].lstrip(".").lower(),
                "page_count": _metadata_value(loader_meta.get("page_count")),
                "pdf_title": _metadata_value(loader_meta.get("pdf_title")),
                "source_kind": "url" if source_url else "upload",
                "stored_at": datetime.now().astimezone().isoformat(),
            }
            docs = []
            total_chunks = len([c for c in chunks if c.strip()])
            for chunk_index, chunk in enumerate((c for c in chunks if c.strip()), start=1):
                chunk_meta = dict(base_meta)
                chunk_meta.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_count": total_chunks,
                        "chunk_chars": len(chunk),
                        "section": _section_for_chunk(chunk),
                        "page_range": _page_range_for_chunk(chunk),
                    }
                )
                docs.append(Document(page_content=chunk, metadata=chunk_meta))
            if not docs:
                logger.info("Insertion store skipped file=%s reason=no_non_empty_chunks", os.path.basename(file_path))
                return {"stored": 0, "message": "No non-empty chunks to store."}
            total_chars = sum(len(d.page_content) for d in docs)
            logger.info(
                "Insertion store start file=%s loader=%s chunks=%d text_chars=%d collection=%s",
                os.path.basename(file_path),
                loader_meta.get("loader", ""),
                len(docs),
                total_chars,
                INSERTION_COLLECTION,
            )
            for idx, doc in enumerate(docs, start=1):
                logger.info(
                    "Insertion store chunk file=%s chunk=%d/%d chars=%d",
                    os.path.basename(file_path),
                    idx,
                    len(docs),
                    len(doc.page_content),
                )
            vector_store.add_documents(docs)
            if run_record is not None:
                run_record["metrics"]["stored_chunks"] = len(docs)
                run_record["metrics"]["stored_text_chars"] = total_chars
                metrics.log_event(
                    run_record,
                    "stored",
                    chunks=len(docs),
                    text_chars=total_chars,
                    collection=INSERTION_COLLECTION,
                )
            logger.info(
                "Insertion store complete file=%s stored_chunks=%d text_chars=%d collection=%s",
                os.path.basename(file_path),
                len(docs),
                total_chars,
                INSERTION_COLLECTION,
            )
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
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6, return_intermediate_steps=True)

    # --- public entrypoint ---
    def ingest_file(self, file_path: str, *, force_vision: bool = False, source_url: Optional[str] = None) -> dict:
        """Run extraction + agent over a single file. Returns a result dict."""
        filename = os.path.basename(file_path)
        with metrics.track_run(
            "insertion",
            action="ingest",
            metadata={"filename": filename, "path": file_path, "model": insertion_agent_model()},
        ) as run:
            ingest_start = time.perf_counter()
            logger.info("Insertion ingest start file=%s path=%s source_url=%s", filename, file_path, source_url or "")
            load_result = self.extract_text(file_path, run_record=run)
            text = load_result.text.strip()
            pages_inserted = _pages_inserted_from_metadata(load_result.metadata)
            if not text:
                run["status"] = "error"
                run["error"] = "No text extracted from file."
                logger.error("Insertion ingest failed file=%s reason=no_text", filename)
                raise RuntimeError("No text extracted from file.")

            excerpt = text[:4000]
            executor = self._build_agent(
                file_path,
                text,
                load_result.metadata,
                source_url=source_url,
                run_record=run,
            )
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
                logger.warning("Insertion agent fallback file=%s error=%s", filename, e)
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = splitter.split_text(text)
                docs = [
                    Document(
                        page_content=c,
                        metadata={
                            "source": filename,
                            "source_path": file_path,
                            "source_url": source_url or "",
                            "loader": load_result.metadata.get("loader", ""),
                            "title": filename,
                            "subject": "",
                            "section": _section_for_chunk(c),
                            "page_range": _page_range_for_chunk(c),
                            "document_type": os.path.splitext(file_path)[1].lstrip(".").lower(),
                            "fallback": True,
                        },
                    )
                    for c in chunks
                    if c.strip()
                ]
                self.get_vector_store().add_documents(docs)
                fallback_chars = sum(len(d.page_content) for d in docs)
                agent_output = f"Agent error ({e}); stored {len(docs)} chunks directly."
                run["metrics"]["stored_chunks"] = len(docs)
                run["metrics"]["chars"] = len(text)
                run["metrics"]["agent_fallback"] = True
                logger.info(
                    "Insertion fallback store complete file=%s stored_chunks=%d text_chars=%d elapsed_s=%.2f",
                    filename,
                    len(docs),
                    fallback_chars,
                    time.perf_counter() - ingest_start,
                )
                return {
                    "file": filename,
                    "path": file_path,
                    "stored_chunks": len(docs),
                    "text_chars": len(text),
                    "loader": load_result.metadata.get("loader", ""),
                    "pages_inserted": pages_inserted,
                    "page_count": load_result.metadata.get("page_count"),
                    "agent_output": agent_output,
                    "fallback": True,
                    "warnings": load_result.warnings,
                    "run_id": run["id"],
                    "source_url": source_url,
                }

            # The store tool writes the authoritative count into the run record.
            # Keep intermediate-step parsing only as a compatibility fallback.
            stored_chunks = int(run["metrics"].get("stored_chunks") or 0)
            intermediate = output.get("intermediate_steps") if isinstance(output, dict) else None
            if intermediate:
                for _, obs in intermediate:
                    if isinstance(obs, dict) and "stored" in obs:
                        stored_chunks = max(stored_chunks, int(obs.get("stored") or 0))
            run["metrics"]["stored_chunks"] = stored_chunks
            run["metrics"]["chars"] = len(text)
            logger.info(
                "Insertion ingest complete file=%s stored_chunks=%d text_chars=%d loader=%s elapsed_s=%.2f",
                filename,
                stored_chunks,
                len(text),
                load_result.metadata.get("loader", ""),
                time.perf_counter() - ingest_start,
            )
            return {
                "file": filename,
                "path": file_path,
                "stored_chunks": stored_chunks,
                "text_chars": len(text),
                "loader": load_result.metadata.get("loader", ""),
                "pages_inserted": pages_inserted,
                "page_count": load_result.metadata.get("page_count"),
                "agent_output": agent_output,
                "fallback": False,
                "warnings": load_result.warnings,
                "run_id": run["id"],
                "source_url": source_url,
            }


_SINGLETON: Optional[InsertionAgent] = None


def get_insertion_agent() -> InsertionAgent:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = InsertionAgent()
    return _SINGLETON
