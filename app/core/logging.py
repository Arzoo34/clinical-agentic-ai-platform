"""Structured JSON logger for pipeline components."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    """Structured JSON logger that writes to stdout and optional JSONL file."""

    def __init__(
        self,
        name: str,
        log_file: Optional[Path | str] = None,
        level: int = logging.INFO,
    ):
        self.name = name
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self._console_logger = logging.getLogger(f"platform.{name}")
        self._console_logger.setLevel(level)
        if not self._console_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
            )
            self._console_logger.addHandler(handler)

    def _log(
        self,
        level_name: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": self.name,
            "level": level_name,
            "message": message,
            "context": context or {},
        }

        # Console log
        log_fn = getattr(self._console_logger, level_name.lower(), self._console_logger.info)
        ctx_str = f" | {json.dumps(context, default=str)}" if context else ""
        log_fn(f"{message}{ctx_str}")

        # JSONL file log
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception:
                pass

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self._log("DEBUG", message, context)

    def info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self._log("INFO", message, context)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self._log("WARNING", message, context)

    def error(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self._log("ERROR", message, context)
