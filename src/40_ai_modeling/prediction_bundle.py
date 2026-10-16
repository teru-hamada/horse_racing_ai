# Actionsへ渡す学習済みモデルと推論用DBのスナップショットを生成・検証する。
# 成功済みtop3モデルの最新版を選び、必要なファイルがなければ古いモデルへ切り替えず失敗させる。
# アーカイブ、依存ライブラリ一覧、各展開ファイルのSHA-256とモデルIDを確認してから利用する。
#
"""Export/verify a versioned runtime snapshot for manual Actions predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import shutil
import tempfile
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path

import duckdb
from .common.calibration import load_calibration

# プロジェクト既定の保存先。DBを差し替えるテストではこの参照も隔離先へ変更する。
PATHS = import_module("src.00_common.config").PATHS
# バンドルへ収録する必須のモデル成果物。どれか一つでも欠ければ生成を止める。
MODEL_FILES = ("model.pt", "preprocessor.joblib", "metrics.json", "manifest.json")
# ZIPに存在してよいファイル名の集合。展開前に過不足と重複を検査する。
MEMBERS = {"racing.duckdb", "features.duckdb", *(f"model/{name}" for name in MODEL_FILES)}
CALIBRATED_MEMBERS = MEMBERS | {"model/calibration.json"}
# 実行環境を固定・確認するライブラリ名の一覧。
PACKAGES = ("numpy", "pandas", "requests", "beautifulsoup4", "lxml", "duckdb",
            "pyarrow", "torch", "scikit-learn", "joblib")


# 大きなZIPやDBを一度にメモリへ載せず、1 MiBずつ読み進めてSHA-256を計算する。
def sha256(path: Path) -> str:
    # ファイルや入力データのハッシュを順次更新するオブジェクト。全体を一度に保持しない。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# completedかつtop3の登録だけを対象に、作成日時の新しいモデルを選択する。
def latest_model(connection) -> dict:
    rows = connection.execute(
        "SELECT model_run_id, created_at, model_path FROM model_runs "
        "WHERE status = 'completed' AND task_name = 'top3' "
        "ORDER BY created_at DESC NULLS LAST, model_run_id DESC LIMIT 1"
    ).fetchall()
    # 成功済みTop3モデルが一つもなければ、未学習としてバンドル生成を止める。
    if not rows:
        raise ValueError("学習成功済みのtop3モデルが登録されていません。")
    model_id, created_at, model_path = rows[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
        raise ValueError("Invalid model ID")
    return {"model_id": model_id, "created_at": str(created_at), "model_path": model_path}


# 元DBを読み取り専用で開き、必要な表とモデルを作業領域へ複製してから4つの配布ファイルへ反映する。
def export_bundle(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary, ExitStack() as stack:
        stage = Path(temporary)
        # Hold read-only locks while exporting both DBs; do not modify source DBs.
        racing = stack.enter_context(duckdb.connect(str(PATHS.database), read_only=True))
        features = stack.enter_context(duckdb.connect(str(PATHS.feature_database), read_only=True))
        selected = latest_model(racing)
        model_dir = Path(selected["model_path"])
        if not model_dir.is_dir():
            model_dir = PATHS.models / "top3" / selected["model_id"]
        (stage / "model").mkdir()
        for name in MODEL_FILES:
            source = model_dir / name
            # 最新版の必須成果物が欠けた場合は、古いモデルへ自動切替せず不足を通知する。
            if not source.is_file():
                raise FileNotFoundError(f"最新モデルの必須ファイルがありません: {source}")
            shutil.copy2(source, stage / "model" / name)
        metrics = json.loads((stage / "model/metrics.json").read_text(encoding="utf-8"))
        # 旧モデルは形式1を維持。補正定義を持つモデルは形式2にして必ずハッシュ検証する。
        calibration_path = model_dir / "calibration.json"
        members = CALIBRATED_MEMBERS if calibration_path.exists() else MEMBERS
        if calibration_path.exists():
            shutil.copy2(calibration_path, stage / "model/calibration.json")
        load_calibration(stage / "model", metrics)
        model_manifest = json.loads((stage / "model/manifest.json").read_text(encoding="utf-8"))
        if any(item.get("model_run_id") != selected["model_id"] for item in (metrics, model_manifest)):
            raise ValueError("DBとモデルのIDが一致しません。")
        history_rows = racing.execute("SELECT count(*) FROM race_records WHERE dataset_type='historical'").fetchone()[0]
        speed_rows = features.execute("SELECT count(*) FROM performance_features WHERE performance_feature_name='speed_index' AND performance_feature_version='1.0.0'").fetchone()[0]
        # 予想に必要な過去結果と速度指数の両方があることを確認し、欠けたバンドルを配布しない。
        if not history_rows or not speed_rows:
            raise ValueError("過去レースまたは速度指数データがありません。")
        # Export only inference inputs, avoiding unused training feature tables.
        for source, filename, tables in (
            (racing, "racing.duckdb", {
                "race_records": "SELECT * FROM race_records WHERE dataset_type='historical'",
                "model_runs": "SELECT * FROM model_runs",
            }),
            (features, "features.duckdb", {
                "performance_features": "SELECT * FROM performance_features WHERE performance_feature_name='speed_index' AND performance_feature_version='1.0.0'",
            }),
        ):
            with duckdb.connect(str(stage / filename)) as destination:
                for table, query in tables.items():
                    frame = source.execute(query).df()
                    destination.register("payload", frame)
                    destination.execute(f"CREATE TABLE {table} AS SELECT * FROM payload")
                    destination.unregister("payload")
        versions = {name: metadata.version(name) for name in PACKAGES}
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        (stage / ".python-version").write_text(python_version + "\n", encoding="utf-8", newline="\n")
        # CPU wheels use a local suffix that need not be portable across hosts.
        requirements = "".join(f"{name}=={version.split('+')[0]}\n" for name, version in versions.items())
        (stage / "requirements.txt").write_text(requirements, encoding="utf-8", newline="\n")
        with zipfile.ZipFile(stage / "runtime.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted(members):
                archive.write(stage / name, name)
        manifest = {
            "format": 2 if members == CALIBRATED_MEMBERS else 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "python_version": python_version,
            "model_id": selected["model_id"], "model_created_at": selected["created_at"],
            "history_rows": history_rows, "speed_rows": speed_rows, "versions": versions,
            "files": {name: sha256(stage / name) for name in sorted(members)},
            "archive_sha256": sha256(stage / "runtime.zip"),
            "requirements_sha256": sha256(stage / "requirements.txt"),
            "archive_bytes": (stage / "runtime.zip").stat().st_size,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        for name in ("runtime.zip", "requirements.txt", "manifest.json", ".python-version"):
            (stage / name).replace(output / name)
    return manifest


# ZIP全体のハッシュとメンバー一覧を先に検査し、展開後も各ファイルのハッシュとモデルIDを確認する。
def unpack_bundle(bundle: Path, output: Path) -> dict:
    # bundle: runtime.zipとmanifestなどがある配布用バンドルのディレクトリ。
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    # 対応する形式と収録ファイル一覧を確認し、未知の構成を展開しない。
    members = {1: MEMBERS, 2: CALIBRATED_MEMBERS}.get(manifest.get("format"))
    if members is None or set(manifest["files"]) != members:
        raise ValueError("Unsupported bundle manifest")
    for name, key in (("runtime.zip", "archive_sha256"), ("requirements.txt", "requirements_sha256")):
        # マニフェストと異なるZIPや依存一覧は、モデル読み込みより前に拒否する。
        if sha256(bundle / name) != manifest[key]:
            raise ValueError(f"Bundle hash mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(bundle / "runtime.zip") as archive:
        # 余分なファイルや同名重複を拒否し、期待する固定メンバーだけを展開する。
        if set(archive.namelist()) != members or len(archive.namelist()) != len(members):
            raise ValueError("Unexpected archive members")
        for name in sorted(members):
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            # アーカイブ全体だけでなく、展開済みの各ファイルについても内容一致を確認する。
            if sha256(destination) != manifest["files"][name]:
                raise ValueError(f"Payload hash mismatch: {name}")
    with duckdb.connect(str(output / "racing.duckdb"), read_only=True) as con:
        selected = latest_model(con)
    metrics = json.loads((output / "model/metrics.json").read_text(encoding="utf-8"))
    load_calibration(output / "model", metrics)
    model_manifest = json.loads((output / "model/manifest.json").read_text(encoding="utf-8"))
    if (selected["model_id"] != manifest["model_id"]
            or any(item.get("model_run_id") != selected["model_id"] for item in (metrics, model_manifest))):
        raise ValueError("最新モデル登録とパッケージのモデルIDが一致しません。")
    return manifest


# モジュールとしてimportしただけでは実行せず、CLIとして起動された場合だけ入口を呼ぶ。
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("prediction_bundle"))
    print(json.dumps(export_bundle(parser.parse_args().output), ensure_ascii=False, indent=2))
