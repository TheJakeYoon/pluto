"""Agent configuration: resolve model names from backend/settings.json."""

from __future__ import annotations

import json
import os
from typing import List

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(_BASE_DIR, "settings.json")
PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
GENERATED_DIR = os.path.join(DATA_DIR, "generated")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "database", "chroma")
AGENT_LOG_DIR = os.path.join(_BASE_DIR, "agents", "logs")

for _d in (DATA_DIR, UPLOADS_DIR, GENERATED_DIR, AGENT_LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# Shared Chroma collection used by RAG text, insertion uploads/URLs, and storage browsing.
INSERTION_COLLECTION = os.environ.get("AGENT_INSERTION_COLLECTION") or os.environ.get("RAG_COLLECTION") or "agent_kb"


def _load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _first_string(values) -> str | None:
    if isinstance(values, list):
        for v in values:
            if isinstance(v, str) and v.strip():
                return v.strip()
    elif isinstance(values, str) and values.strip():
        return values.strip()
    return None


def insertion_agent_model() -> str:
    s = _load_settings()
    return (
        _first_string(s.get("insertion_agent"))
        or _first_string(s.get("local_models"))
        or "gemma4:26b"
    )


def education_agent_model() -> str:
    s = _load_settings()
    return (
        _first_string(s.get("education_agent"))
        or _first_string(s.get("local_models"))
        or "gemma4:26b"
    )


def embedding_model() -> str:
    s = _load_settings()
    return (
        _first_string(s.get("embedding_model"))
        or os.environ.get("RAG_EMBEDDING_MODEL")
        or "qwen3-embedding:8b"
    )


def allowed_insertion_extensions() -> List[str]:
    return [
        ".pdf",
        ".txt",
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".json",
        ".html",
        ".htm",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".md",
    ]


def is_allowed_filename(name: str) -> bool:
    if not name:
        return False
    lower = name.lower()
    return any(lower.endswith(ext) for ext in allowed_insertion_extensions())
