from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable


LogCallback = Callable[[str], None]


class AppLogger:
    """File logger that can also stream formatted lines into the Streamlit UI."""

    def __init__(self, log_dir: Path, callback: LogCallback | None = None) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"app_{datetime.now():%Y%m%d}.log"
        self.callback = callback

        self.logger = logging.getLogger(f"horse_racing_ai.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.handlers.clear()
        self.logger.addHandler(handler)

    def _emit(self, level: str, message: str) -> None:
        getattr(self.logger, level.lower())(message)
        line = f"[{datetime.now():%H:%M:%S}] {level.upper():7s} {message}"
        if self.callback:
            self.callback(line)

    def info(self, message: str) -> None:
        self._emit("info", message)

    def warning(self, message: str) -> None:
        self._emit("warning", message)

    def error(self, message: str) -> None:
        self._emit("error", message)

    def exception(self, message: str) -> None:
        self.logger.exception(message)
        line = f"[{datetime.now():%H:%M:%S}] ERROR   {message}"
        if self.callback:
            self.callback(line)
