"""Lightweight persistence + in-memory ring buffer for agent runs.

Every agent run appends a JSONL record to ``backend/agents/logs/<agent>.jsonl``
and is kept in a capped in-memory deque for the stats endpoint.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from typing import Deque, Dict, List, Optional

from .config import AGENT_LOG_DIR

_MAX_MEMORY_RUNS = 500
_lock = threading.Lock()
_runs: Dict[str, Deque[dict]] = {
    "insertion": deque(maxlen=_MAX_MEMORY_RUNS),
    "education": deque(maxlen=_MAX_MEMORY_RUNS),
}


def _log_path(agent: str) -> str:
    os.makedirs(AGENT_LOG_DIR, exist_ok=True)
    return os.path.join(AGENT_LOG_DIR, f"{agent}.jsonl")


def _append_jsonl(agent: str, record: dict) -> None:
    path = _log_path(agent)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # metrics must never break agent runs


def _load_from_disk(agent: str) -> List[dict]:
    path = _log_path(agent)
    if not os.path.isfile(path):
        return []
    out: List[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def _ensure_bootstrapped(agent: str) -> None:
    """Populate the in-memory deque from disk on first access."""
    with _lock:
        if _runs[agent] or not os.path.isfile(_log_path(agent)):
            return
        for rec in _load_from_disk(agent)[-_MAX_MEMORY_RUNS:]:
            _runs[agent].append(rec)


@contextmanager
def track_run(agent: str, *, action: str, metadata: Optional[dict] = None):
    """Context manager that records start/end, duration, outcome, and errors.

    Yields a mutable dict the caller can update (``tokens``, ``output_chars`` …).
    On exit the finalized record is persisted and returned in-memory.
    """
    if agent not in _runs:
        _runs[agent] = deque(maxlen=_MAX_MEMORY_RUNS)
    _ensure_bootstrapped(agent)

    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    started_iso = datetime.now().astimezone().isoformat()
    record: dict = {
        "id": run_id,
        "agent": agent,
        "action": action,
        "status": "running",
        "started_at": started_iso,
        "metadata": dict(metadata or {}),
        "metrics": {},
        "events": [],
    }
    with _lock:
        _runs[agent].append(record)
    try:
        yield record
        record["status"] = record.get("status", "running")
        if record["status"] == "running":
            record["status"] = "ok"
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        record["duration_s"] = round(time.perf_counter() - started, 3)
        record["ended_at"] = datetime.now().astimezone().isoformat()
        _append_jsonl(agent, record)


def log_event(record: dict, event: str, **fields) -> None:
    """Attach a structured event to an in-progress record (visible in the UI)."""
    entry = {"event": event, "t": datetime.now().astimezone().isoformat(), **fields}
    record.setdefault("events", []).append(entry)


def recent_runs(agent: Optional[str] = None, *, limit: int = 50) -> List[dict]:
    _ensure_bootstrapped("insertion")
    _ensure_bootstrapped("education")
    with _lock:
        if agent:
            items = list(_runs.get(agent, deque()))
        else:
            items = list(_runs["insertion"]) + list(_runs["education"])
    items.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return items[:limit]


def stats(agent: Optional[str] = None) -> dict:
    runs = recent_runs(agent, limit=_MAX_MEMORY_RUNS)
    total = len(runs)
    ok = sum(1 for r in runs if r.get("status") == "ok")
    err = sum(1 for r in runs if r.get("status") == "error")
    running = sum(1 for r in runs if r.get("status") == "running")
    durations = [r.get("duration_s") for r in runs if isinstance(r.get("duration_s"), (int, float))]
    avg = round(sum(durations) / len(durations), 3) if durations else 0.0
    last = runs[0] if runs else None
    actions: Dict[str, int] = {}
    for r in runs:
        actions[r.get("action", "unknown")] = actions.get(r.get("action", "unknown"), 0) + 1
    return {
        "agent": agent or "all",
        "total_runs": total,
        "ok_runs": ok,
        "error_runs": err,
        "running_runs": running,
        "success_rate": round((ok / total) * 100, 2) if total else 0.0,
        "avg_duration_s": avg,
        "actions": actions,
        "last_run": last,
    }


def evaluate_run(run_id: str, agent: str, score: Optional[float], feedback: Optional[str]) -> Optional[dict]:
    """Attach a human evaluation (score / feedback) to a run and persist it."""
    _ensure_bootstrapped(agent)
    with _lock:
        bucket = _runs.get(agent)
        target = None
        if bucket:
            for rec in bucket:
                if rec.get("id") == run_id:
                    target = rec
                    break
        if target is None:
            return None
        target.setdefault("evaluation", {})
        if score is not None:
            target["evaluation"]["score"] = float(score)
        if feedback is not None:
            target["evaluation"]["feedback"] = feedback
        target["evaluation"]["evaluated_at"] = datetime.now().astimezone().isoformat()
        # Re-append to JSONL so the newest state is recorded
        _append_jsonl(agent, {"id": run_id, "evaluation": target["evaluation"]})
        return dict(target)
