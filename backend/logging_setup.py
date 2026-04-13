"""Rotating file logging for long-running uvicorn processes."""

from __future__ import annotations

import logging
import logging.handlers
import os
from typing import Optional

_configured = False
_shared_handler: Optional[logging.handlers.RotatingFileHandler] = None


def configure_app_logging(base_dir: str) -> None:
    """Attach one shared RotatingFileHandler to uvicorn/fastapi loggers (idempotent per process)."""
    global _configured, _shared_handler
    if _configured:
        return
    _configured = True

    log_dir = os.environ.get("APP_LOG_DIR", os.path.join(base_dir, "logs"))
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, os.environ.get("APP_LOG_FILENAME", "app.log"))
    max_bytes = int(os.environ.get("APP_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    backup_count = int(os.environ.get("APP_LOG_BACKUP_COUNT", "5"))

    _shared_handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    _shared_handler.setLevel(logging.DEBUG)
    _shared_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        log = logging.getLogger(name)
        if _shared_handler not in log.handlers:
            log.addHandler(_shared_handler)
        if log.level == logging.NOTSET:
            log.setLevel(logging.INFO)
