#!/usr/bin/env bash
# Run the backend with the project venv (required for RAG/Chroma).
# Usage: ./run.sh   or   ./run.sh --reload
cd "$(dirname "$0")"
exec ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 "$@"
