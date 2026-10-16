# 学習成果物の構成情報をmanifest.jsonとしてUTF-8で保存する。
# モデルと同じディレクトリへ記録し、タスク・特徴量・学習期間などを後から確認できるようにする。
#
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# モデル構成を日時などの値も文字列化できるJSONへ整え、成果物と同じ場所へ保存する。
def save_manifest(model_dir: Path, manifest: dict[str, Any]) -> Path:
    # model_dir: 重み・前処理器・指標が同じ学習runとして保存されたディレクトリ。
    path = model_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
