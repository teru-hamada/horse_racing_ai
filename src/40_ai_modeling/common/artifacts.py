from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_manifest(model_dir: Path, manifest: dict[str, Any]) -> Path:
    path = model_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
