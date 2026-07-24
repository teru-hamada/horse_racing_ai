from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path
    data: Path
    raw_html: Path
    processed: Path
    models: Path
    logs: Path
    database: Path


def get_paths(root: Path | None = None) -> AppPaths:
    root_path = (root or Path(__file__).resolve().parents[1]).resolve()
    data = root_path / "data"
    paths = AppPaths(
        root=root_path,
        data=data,
        raw_html=data / "raw_html",
        processed=data / "processed",
        models=root_path / "models",
        logs=root_path / "logs",
        database=data / "racing.duckdb",
    )
    for directory in (paths.data, paths.raw_html, paths.processed, paths.models, paths.logs):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


PATHS = get_paths()
