"""Manual one-race inference, or offline replay of an Actions artifact."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import sys
import time
from datetime import date
from importlib import import_module, metadata
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .html_fetch_smoke import run_probe
from .prediction_bundle import PACKAGES, PATHS, latest_model, sha256, unpack_bundle
from .static_site import build_prediction_site

storage = import_module("src.00_common.storage")


def validate_predictions(card: pd.DataFrame, result: pd.DataFrame) -> None:
    keys = ["race_id", "horse_id", "horse_number"]
    if result.empty or result.duplicated(keys).any() or len(result) != len(card):
        raise ValueError("予想頭数の不足または重複があります。")
    for frame in (card, result):
        if frame[keys].isna().any().any():
            raise ValueError("予想対象の識別情報に欠損があります。")
    if set(map(tuple, card[keys].astype(str).values)) != set(map(tuple, result[keys].astype(str).values)):
        raise ValueError("出馬表と予想の対象馬が一致しません。")
    probability = result["top3_probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("予測確率が非有限値または0～1の範囲外です。")


def compare_results(output: Path, reference: Path) -> dict:
    differences = {}
    for name, keys in (("predictions.csv", ["race_id", "horse_id", "horse_number"]),
                       ("bets.csv", ["race_id", "bet_type", "selection"])):
        left, right = [pd.read_csv(path / name, dtype={"race_id": str, "horse_id": str, "selection": str})
                       for path in (output, reference)]
        try:
            pd.testing.assert_frame_equal(
                left.sort_values(keys).reset_index(drop=True),
                right.sort_values(keys).reset_index(drop=True),
                check_dtype=False, check_exact=False, atol=1e-6, rtol=1e-5,
            )
        except AssertionError as exc:
            differences[name] = str(exc)
    return {"status": "ok" if not differences else "error", "differences": differences,
            "absolute_tolerance": 1e-6, "relative_tolerance": 1e-5}


def run_prediction(bundle: Path, output: Path, race_date: date | None = None,
                   race_id: str | None = None, replay: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "error", "race_date": str(race_date), "race_id": race_id}
    report.update(prediction_status="skipped", odds_match_status="skipped", betting_status="skipped")
    logger = logging.getLogger(f"prediction_smoke.{output}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "prediction.log", encoding="utf-8")
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    try:
        manifest = unpack_bundle(bundle, output / "work")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if manifest.get("python_version", python_version) != python_version:
            raise ValueError(f"Python {manifest['python_version']} が必要です（現在: {python_version}）。")
        report.update(model_id=manifest["model_id"], bundle_sha256=manifest["archive_sha256"],
                      python_version=python_version,
                      bundle_exported_at=manifest["exported_at"], platform=platform.platform(),
                      github_sha=os.environ.get("GITHUB_SHA"),
                      versions={name: metadata.version(name) for name in PACKAGES})
        if replay is None:
            # Detect a newer registry pushed without regenerating the model bundle.
            with duckdb.connect(str(PATHS.database), read_only=True) as con:
                registered = latest_model(con)
            if registered["model_id"] != manifest["model_id"]:
                raise ValueError("登録DBの最新モデルとパッケージが異なります。パッケージを再生成してください。")
        logger.info("使用モデル: %s / bundle: %s", manifest["model_id"], manifest["archive_sha256"])
        for name in ("scikit-learn", "numpy", "pandas", "torch", "joblib", "duckdb"):
            if report["versions"][name].split("+")[0] != manifest["versions"][name].split("+")[0]:
                raise ValueError(f"パッケージの実行環境とバージョンが異なります: {name}")
        if replay is not None:
            original = json.loads((replay / "prediction_report.json").read_text(encoding="utf-8"))
            if original["status"] != "ok" or original["bundle_sha256"] != manifest["archive_sha256"]:
                raise ValueError("再現元が正常終了していないか、実行用パッケージが異なります。")
            race_date = date.fromisoformat(original["race_date"])
            race_id = original["race_id"]
            (output / "fetch").mkdir()
            source = replay / "fetch/smoke.duckdb"
            if sha256(source) != original["input_database_sha256"]:
                raise ValueError("再現用DBのハッシュが一致しません。")
            shutil.copy2(source, output / "fetch/smoke.duckdb")
        else:
            probe = run_probe(race_date, race_id, output / "fetch", verify_db=True)
            report["fetch_status"] = probe["status"]
            if probe["status"] != "ok":
                report["status"] = probe["status"]
                raise ValueError("HTML取得・DB照合が未合格です。fetch/report.jsonを確認してください。")
        report.update(race_date=race_date.isoformat(), race_id=race_id)
        database = output / "fetch/smoke.duckdb"
        # Read only: the artifact DB hash is stable for local replay.
        with duckdb.connect(str(database), read_only=True) as con:
            card = con.execute("SELECT * FROM race_records WHERE dataset_type='upcoming' ORDER BY horse_number").df()
            odds = con.execute("SELECT * FROM race_odds ORDER BY bet_type,selection_1,selection_2,selection_3").df()
        report["input_database_sha256"] = sha256(database)
        if not card["race_id"].eq(race_id).all() or not pd.to_datetime(card["race_date"]).dt.date.eq(race_date).all():
            raise ValueError("入力DBと予想対象が一致しません。")
        with duckdb.connect(str(output / "work/racing.duckdb"), read_only=True) as con:
            history = con.execute("SELECT * FROM race_records WHERE dataset_type='historical' AND race_date < ?", [race_date]).df()
        if history.empty:
            raise ValueError("予想対象日より前の過去レースがありません。")
        report["history_rows"] = len(history)
        report["history_latest_date"] = str(history["race_date"].max().date())
        import torch
        torch.set_num_threads(1)
        model = import_module("src.30_ai_modeling.tasks.top3.model")
        report["prediction_status"] = "error"
        predictions, metrics = model.infer_race(history, card, race_id, output / "work/model",
                                                feature_database=output / "work/features.duckdb")
        if metrics["model_run_id"] != report["model_id"]:
            raise ValueError("使用モデルIDが一致しません。")
        validate_predictions(card, predictions)
        report["prediction_status"] = "ok"
        predictions["model_run_id"] = report["model_id"]
        predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
        # Independently confirm the odds join before recommendation thresholds filter bets.
        win = odds[odds.bet_type.eq("win")]
        report["odds_match_status"] = "error"
        joined = predictions.merge(win[["race_id", "selection_1", "odds_min"]],
                                   left_on=["race_id", "horse_number"], right_on=["race_id", "selection_1"],
                                   how="left", validate="one_to_one")
        if not joined.odds_min.gt(0).all():
            raise ValueError("予想と単勝オッズの照合に失敗しました。")
        joined.to_csv(output / "prediction_odds.csv", index=False, encoding="utf-8-sig")
        report["odds_match_status"] = "ok"
        report["betting_status"] = "error"
        bets = import_module("src.30_ai_modeling.betting").calculate_bet_recommendations(predictions, odds, best_only=False)
        if not bets.empty:
            numeric = bets[["estimated_probability", "odds_used", "recovery_rate_percent"]].to_numpy(dtype=float)
            if not np.isfinite(numeric).all() or not bets.estimated_probability.between(0, 1).all() or not bets.odds_used.gt(0).all():
                raise ValueError("買い目計算結果が不正です。")
        bets.to_csv(output / "bets.csv", index=False, encoding="utf-8-sig")
        report["betting_status"] = "ok"
        build_prediction_site(predictions, race_date, report["model_id"], output / "preview", bet_recommendations=bets)
        report.update(prediction_rows=len(predictions), matched_win_runners=len(joined), bet_rows=len(bets),
                      bet_note="推奨条件を満たす買い目なし" if bets.empty else "計算成功", status="ok")
        if replay is not None:
            report["comparison"] = compare_results(output, replay)
            if report["comparison"]["status"] != "ok":
                report["status"] = "error"
        logger.info("検証結果: %s", json.dumps(report, ensure_ascii=False))
    except Exception as exc:
        report["error"] = str(exc)
        if report["status"] != "incomplete":
            report["status"] = "error"
        logger.exception("予想テストに失敗")
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        (output / "prediction_report.json").write_text(text, encoding="utf-8")
        summary = "## 1レース予想テスト\n\n```json\n" + text + "\n```\n"
        (output / "summary.md").write_text(summary, encoding="utf-8")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(summary)
        for item in list(logger.handlers):
            item.close()
            logger.removeHandler(item)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("prediction_bundle"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--race-date", type=date.fromisoformat)
    parser.add_argument("--race-id")
    parser.add_argument("--replay", type=Path, help="artifact directory; uses saved inputs without network")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("出力先には新しいフォルダを指定してください。")
    if not args.replay:
        if not args.race_date or not re.fullmatch(r"[0-9]{12}", args.race_id or ""):
            parser.error("開催日と12桁のrace-idが必要です。")
        if (args.race_id[:4] != str(args.race_date.year) or not 1 <= int(args.race_id[4:6]) <= 10
                or any(not 1 <= int(args.race_id[i:i+2]) <= 12 for i in (6, 8, 10))):
            parser.error("race-idと開催年・開催番号が不正です。")
    elif args.race_date or args.race_id:
        parser.error("replayでは対象日とrace-idを再現元から読み込みます。")
    result = run_prediction(args.bundle, args.output, args.race_date, args.race_id, args.replay)
    return {"ok": 0, "incomplete": 2, "error": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
