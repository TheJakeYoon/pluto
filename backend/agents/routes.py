"""FastAPI router for the insertion + education agents and their metrics."""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import metrics
from .config import (
    GENERATED_DIR,
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


# ---------- config ----------


@router.get("/config")
async def agents_config():
    """Return model assignments, allowed extensions, and output directories."""
    return {
        "insertion_agent_model": insertion_agent_model(),
        "education_agent_model": education_agent_model(),
        "embedding_model": embedding_model(),
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

    for upload in files:
        name = upload.filename or ""
        if not is_allowed_filename(name):
            rejected.append(name)
            continue
        data = await upload.read()
        if not data:
            rejected.append(name)
            continue
        try:
            path = agent.persist_upload(name, data)
        except ValueError:
            rejected.append(name)
            continue

        try:
            result = await asyncio.to_thread(agent.ingest_file, path)
            results.append(result)
        except Exception as e:
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
