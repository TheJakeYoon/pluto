"""FastAPI router for the insertion + education agents and their metrics."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import List, Optional
from urllib.parse import unquote, urlparse

import httpx
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

from . import metrics
from .config import (
    GENERATED_DIR,
    INSERTION_COLLECTION,
    UPLOADS_DIR,
    allowed_insertion_extensions,
    education_agent_model,
    embedding_model,
    insertion_agent_model,
    is_allowed_filename,
)
from .education_agent import get_education_agent
from .insertion_agent import get_insertion_agent

router = APIRouter(prefix="/api/agents", tags=["agents"])
logger = logging.getLogger("uvicorn.error")

_URL_INSERT_MAX_BYTES = 30 * 1024 * 1024
_URL_INSERT_TIMEOUT = httpx.Timeout(20.0, read=60.0)
_URL_EXT_BY_CONTENT_TYPE = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
}


def _filename_from_url(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    host = re.sub(r"[^A-Za-z0-9._-]+", "-", parsed.netloc).strip("-") or "web"
    path_name = os.path.basename(unquote(parsed.path or "")).strip()
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", path_name).strip("-")
    ext = os.path.splitext(safe_name)[1].lower()
    base = os.path.splitext(safe_name)[0].strip("-") or "document"
    if ext not in (".pdf", ".html", ".htm"):
        ctype = (content_type or "").split(";", 1)[0].strip().lower()
        ext = _URL_EXT_BY_CONTENT_TYPE.get(ctype, ".html")
    return f"{host}-{base}{ext}"


async def _download_insertable_url(url: str) -> tuple[str, bytes, str]:
    logger.info("Insertion URL download start url=%s", url)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            async with client.stream("GET", url, timeout=_URL_INSERT_TIMEOUT) as response:
                if response.status_code >= 400:
                    raise HTTPException(status_code=400, detail=f"URL returned HTTP {response.status_code}: {url}")
                ctype = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
                parsed_ext = os.path.splitext(urlparse(str(response.url)).path)[1].lower()
                if ctype not in _URL_EXT_BY_CONTENT_TYPE and parsed_ext not in (".pdf", ".html", ".htm"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"URL must point to a PDF or HTML page (got content-type {ctype or 'unknown'}).",
                    )
                logger.info(
                    "Insertion URL download response url=%s final_url=%s status=%d content_type=%s",
                    url,
                    str(response.url),
                    response.status_code,
                    ctype or "unknown",
                )
                chunks: list[bytes] = []
                total = 0
                next_log_bytes = 5 * 1024 * 1024
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > _URL_INSERT_MAX_BYTES:
                        raise HTTPException(status_code=400, detail="URL download is too large (max 30 MB).")
                    if total >= next_log_bytes:
                        logger.info("Insertion URL download progress url=%s bytes=%d", url, total)
                        next_log_bytes += 5 * 1024 * 1024
                    chunks.append(chunk)
                data = b"".join(chunks)
                if not data:
                    raise HTTPException(status_code=400, detail=f"URL returned an empty body: {url}")
                filename = _filename_from_url(str(response.url), ctype)
                logger.info(
                    "Insertion URL download complete url=%s final_url=%s filename=%s bytes=%d",
                    url,
                    str(response.url),
                    filename,
                    len(data),
                )
                return filename, data, str(response.url)
        except HTTPException:
            raise
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"Could not fetch URL {url}: {e}") from e


# ---------- config ----------


@router.get("/config")
async def agents_config():
    """Return model assignments, allowed extensions, and output directories."""
    return {
        "insertion_agent_model": insertion_agent_model(),
        "education_agent_model": education_agent_model(),
        "embedding_model": embedding_model(),
        "vector_collection": INSERTION_COLLECTION,
        "allowed_insertion_extensions": allowed_insertion_extensions(),
        "uploads_dir": UPLOADS_DIR,
        "generated_dir": GENERATED_DIR,
    }


# ---------- insertion ----------


@router.post("/insertion/upload")
async def insertion_upload(files: List[UploadFile] = File(...)):
    """Upload files for the insertion agent.

    Allowed types: pdf, txt, md, jpg, jpeg, png, heic, json, html/htm,
    doc, docx, xls, xlsx, ppt, pptx. Other types are rejected with 400.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    agent = get_insertion_agent()
    results: List[dict] = []
    rejected: List[str] = []
    logger.info("Insertion upload request start files=%d", len(files))

    for upload in files:
        name = upload.filename or ""
        if not is_allowed_filename(name):
            rejected.append(name)
            logger.warning("Insertion upload rejected file=%s reason=unsupported_extension", name)
            continue
        data = await upload.read()
        if not data:
            rejected.append(name)
            logger.warning("Insertion upload rejected file=%s reason=empty_file", name)
            continue
        try:
            logger.info("Insertion upload file accepted file=%s bytes=%d", name, len(data))
            path = agent.persist_upload(name, data)
        except ValueError:
            rejected.append(name)
            logger.warning("Insertion upload rejected file=%s reason=persist_validation", name)
            continue

        try:
            result = await asyncio.to_thread(agent.ingest_file, path)
            results.append(result)
            logger.info(
                "Insertion upload file complete file=%s stored_chunks=%s text_chars=%s loader=%s",
                name,
                result.get("stored_chunks"),
                result.get("text_chars"),
                result.get("loader"),
            )
        except Exception as e:
            logger.exception("Insertion upload file failed file=%s path=%s", name, path)
            results.append(
                {"file": name, "path": path, "error": str(e), "stored_chunks": 0}
            )

    if not results and rejected:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No allowed files. Supported: "
                + ", ".join(allowed_insertion_extensions()),
                "rejected": rejected,
            },
        )

    logger.info(
        "Insertion upload request complete processed=%d rejected=%d stored_chunks=%d text_chars=%d",
        len(results),
        len(rejected),
        sum(int(r.get("stored_chunks") or 0) for r in results),
        sum(int(r.get("text_chars") or 0) for r in results),
    )
    return {
        "status": "ok",
        "processed": len(results),
        "rejected": rejected,
        "results": results,
    }


class InsertionUrlRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., min_length=1, max_length=10)


@router.post("/insertion/url")
async def insertion_url(req: InsertionUrlRequest):
    """Insert PDF or HTML web links into the knowledge base."""
    agent = get_insertion_agent()
    results: List[dict] = []
    rejected: List[str] = []
    logger.info("Insertion URL request start urls=%d", len(req.urls))

    for raw_url in req.urls:
        url = str(raw_url)
        try:
            filename, data, final_url = await _download_insertable_url(url)
            path = agent.persist_upload(filename, data)
            result = await asyncio.to_thread(agent.ingest_file, path, source_url=final_url)
            results.append(result)
            logger.info(
                "Insertion URL complete url=%s final_url=%s file=%s stored_chunks=%s text_chars=%s loader=%s",
                url,
                final_url,
                filename,
                result.get("stored_chunks"),
                result.get("text_chars"),
                result.get("loader"),
            )
        except HTTPException as e:
            rejected.append(f"{url} — {e.detail}")
            logger.warning("Insertion URL rejected url=%s detail=%s", url, e.detail)
        except Exception as e:
            rejected.append(f"{url} — {e}")
            logger.exception("Insertion URL failed url=%s", url)

    if not results and rejected:
        logger.warning("Insertion URL request rejected all URLs: %s", rejected)
        raise HTTPException(status_code=400, detail={"message": "No URLs could be inserted.", "rejected": rejected})

    logger.info(
        "Insertion URL request complete processed=%d rejected=%d stored_chunks=%d text_chars=%d",
        len(results),
        len(rejected),
        sum(int(r.get("stored_chunks") or 0) for r in results),
        sum(int(r.get("text_chars") or 0) for r in results),
    )
    return {
        "status": "ok",
        "processed": len(results),
        "rejected": rejected,
        "results": results,
    }


@router.get("/insertion/allowed-extensions")
async def insertion_allowed_extensions():
    return {"extensions": allowed_insertion_extensions()}


# ---------- education ----------


class EducationGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=2, description="Topic for the educational content.")
    audience: str = Field("intermediate learners", description="Target audience description.")
    format: str = Field("markdown", description="Output format: markdown, html, json, or pdf.")
    use_web: bool = Field(True, description="Allow the agent to perform web searches.")
    extra_instructions: str = Field("", description="Additional prompts (tone, length, etc.).")


@router.post("/education/generate")
async def education_generate(req: EducationGenerateRequest):
    agent = get_education_agent()
    try:
        result = await asyncio.to_thread(
            agent.generate,
            topic=req.topic,
            audience=req.audience,
            fmt=req.format,
            use_web=req.use_web,
            extra_instructions=req.extra_instructions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@router.get("/education/download")
async def education_download(path: str = Query(..., description="Path returned by /education/generate")):
    abs_path = os.path.abspath(path)
    generated_root = os.path.abspath(GENERATED_DIR)
    if not abs_path.startswith(generated_root + os.sep) and abs_path != generated_root:
        raise HTTPException(status_code=400, detail="Path is outside the generated directory.")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(abs_path, filename=os.path.basename(abs_path))


@router.get("/education/outputs")
async def education_outputs():
    """List generated files so users can download past runs."""
    items: List[dict] = []
    if os.path.isdir(GENERATED_DIR):
        for name in sorted(os.listdir(GENERATED_DIR), reverse=True):
            full = os.path.join(GENERATED_DIR, name)
            if os.path.isfile(full) and not name.startswith("."):
                stat = os.stat(full)
                items.append({"name": name, "path": full, "size": stat.st_size, "mtime": stat.st_mtime})
    return {"files": items}


# ---------- metrics / monitoring ----------


@router.get("/runs")
async def list_runs(
    agent: Optional[str] = Query(None, pattern="^(insertion|education)$"),
    limit: int = Query(50, ge=1, le=500),
):
    return {"runs": metrics.recent_runs(agent, limit=limit)}


@router.get("/stats")
async def get_stats(agent: Optional[str] = Query(None, pattern="^(insertion|education)$")):
    if agent:
        return metrics.stats(agent)
    return {
        "insertion": metrics.stats("insertion"),
        "education": metrics.stats("education"),
    }


class EvaluateRunRequest(BaseModel):
    agent: str = Field(..., pattern="^(insertion|education)$")
    run_id: str
    score: Optional[float] = Field(None, ge=0, le=5, description="0-5 quality rating.")
    feedback: Optional[str] = Field(None, description="Free-form evaluation feedback.")


@router.post("/runs/evaluate")
async def evaluate_run(req: EvaluateRunRequest):
    if req.score is None and not (req.feedback and req.feedback.strip()):
        raise HTTPException(status_code=400, detail="Provide score or feedback.")
    updated = metrics.evaluate_run(req.run_id, req.agent, req.score, req.feedback)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Run {req.run_id} not found.")
    return updated
