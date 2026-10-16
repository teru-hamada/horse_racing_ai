# 指定日のレース一覧を取得し、1レース予想を順番に実行して日次成果物へ集約する。
# 個別の失敗で残りのレースを止めず、成功した予想と全レースの診断結果を分けて保存する。
# JSON・CSV・HTML・Actionsサマリーに各段階の状態を残し、不足の原因を追えるようにする。
#
"""Manual date-wide orchestration with durable per-race diagnostics."""
from __future__ import annotations
from importlib import import_module as _import_module

import argparse
from contextlib import contextmanager
from datetime import date
import html
import json
import logging
import os
from pathlib import Path
import re
import time

import pandas as pd

from .race_prediction import run_prediction

DateCollector = _import_module('src.20_scrapers_html_collection.date_collector').DateCollector
build_prediction_site = _import_module('src.60_publication.static_site').build_prediction_site
aggregate_comparison_outputs = _import_module('src.40_ai_modeling.common.calibration').aggregate_comparison_outputs


# 内側の各レースがActionsサマリーへ大量出力しないよう、一時的に環境変数を外して最後に戻す。
@contextmanager
def race_summaries_off():
    # Avoid exceeding Actions' summary size with dozens of nested JSON reports.
    previous = os.environ.pop("GITHUB_STEP_SUMMARY", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["GITHUB_STEP_SUMMARY"] = previous


# 取得・DB・予想の各レポートを一行へまとめる。未実行の段階はskippedとして残す。
def race_row(race_id, report, directory):
    # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
    fetch_path = directory / "fetch/report.json"
    fetch = json.loads(fetch_path.read_text(encoding="utf-8")) if fetch_path.exists() else {}
    database = fetch.get("database", {})
    checks = database.get("checks", {})
    return {
        "race_id": race_id, "status": report["status"],
        "card": fetch.get("card", {}).get("status", "skipped"),
        "jra_odds": fetch.get("jra_odds", {}).get("status", "skipped"),
        "database": database.get("status", "skipped"),
        "prediction": report.get("prediction_status", "skipped"),
        "odds_match": report.get("odds_match_status", "skipped"),
        "betting": report.get("betting_status", "skipped"),
        "runners": database.get("expected_runners", fetch.get("card", {}).get("runners", 0)),
        "odds_rows": fetch.get("jra_odds", {}).get("rows", 0),
        "prediction_rows": report.get("prediction_rows", 0),
        "bet_rows": report.get("bet_rows", 0),
        "model_id": report.get("model_id", ""),
        "reason": "; ".join(filter(None, [report.get("error"),
            fetch.get("card", {}).get("error"), fetch.get("jra_odds", {}).get("error"),
            ", ".join(database.get("failed_checks", [])),
            "単勝オッズ照合不足" if checks.get("win_odds_match_all_runners") is False else ""])),
        "report": f"races/{race_id}/prediction_report.json",
    }


# 途中までの状態もJSONへ書き出し、後続のレースで失敗しても診断を参照できるようにする。
def save_progress(output, report):
    (output / "date_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(report["races"]).to_csv(output / "race_status.csv", index=False, encoding="utf-8-sig")


# 対象日のレースを処理し、成功分をCSVとプレビューへ集約する。全レースの成否は別途保持する。
def run_date(bundle: Path, output: Path, race_date: date, *, calibration_method: str = "none"):
    # bundle: runtime.zipとmanifestなどがある配布用バンドルのディレクトリ。
    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"race_date": str(race_date), "status": "running", "race_ids": [], "races": [],
              "calibration_method": calibration_method}
    logger = logging.getLogger(f"date_probe.{output}")
    logger.setLevel(logging.INFO)
    handlers = [logging.FileHandler(output / "date.log", encoding="utf-8"), logging.StreamHandler()]
    for handler in handlers:
        logger.addHandler(handler)
    predictions, bets = [], []
    save_progress(output, report)
    try:
        collector = DateCollector(logger, output / "discovery")
        try:
            ids = sorted(set(collector.race_ids_for_date(race_date, str(race_date.year), "upcoming", force=True)))
        finally:
            collector.session.close()
        if any(not re.fullmatch(r"[0-9]{12}", value) or value[:4] != str(race_date.year) for value in ids):
            raise ValueError("開催日一覧のレースIDが不正です。")
        report["race_ids"] = ids
        report["races"] = [{"race_id": race_id, "status": "pending", "reason": "未処理"} for race_id in ids]
        save_progress(output, report)
        if not ids:
            report["status"] = "incomplete"
            report["error"] = "レースを検出できませんでした。非開催日または開催一覧取得不良を確認してください。"
        for index, race_id in enumerate(ids, 1):
            logger.info("[%s/%s] %s", index, len(ids), race_id)
            directory = output / "races" / race_id
            try:
                with race_summaries_off():
                    result = run_prediction(bundle, directory, race_date, race_id,
                                            **({"calibration_method": calibration_method} if calibration_method != "none" else {}))
                row = race_row(race_id, result, directory)
                if result["status"] == "ok":
                    race_predictions = pd.read_csv(directory / "predictions.csv", dtype={"race_id": str, "horse_id": str})
                    race_bets = pd.read_csv(directory / "bets.csv", dtype={"race_id": str, "selection": str})
                    predictions.append(race_predictions)
                    bets.append(race_bets)
            # 処理失敗の原因をレポートへ残す。成功扱いや開催なしへの置き換えは行わない。
            except Exception as exc:
                logger.exception("レース処理失敗: %s", race_id)
                row = {"race_id": race_id, "status": "error", "reason": str(exc)}
            report["races"][index - 1] = row
            save_progress(output, report)
        if ids:
            statuses = [row["status"] for row in report["races"]]
            report["status"] = "error" if "error" in statuses else ("ok" if all(s == "ok" for s in statuses) else "incomplete")
        if predictions:
            combined = pd.concat(predictions, ignore_index=True)
            combined_bets = pd.concat(bets, ignore_index=True)
            if combined.model_run_id.nunique() != 1:
                raise ValueError("レース間で使用モデルが異なります。")
            combined.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
            combined_bets.to_csv(output / "bets.csv", index=False, encoding="utf-8-sig")
            aggregate_comparison_outputs(output, [output / 'races' / row['race_id'] for row in report['races'] if row['status'] == 'ok'])
            build_prediction_site(combined, race_date, combined.model_run_id.iloc[0], output / "preview", bet_recommendations=combined_bets)
    # 処理失敗の原因をレポートへ残す。成功扱いや開催なしへの置き換えは行わない。
    except Exception as exc:
        report.update(status="error", error=str(exc))
        logger.exception("開催日テスト失敗")
    finally:
        report["counts"] = {status: sum(r["status"] == status for r in report["races"]) for status in ("ok", "incomplete", "error", "pending")}
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        save_progress(output, report)
        table = pd.DataFrame(report["races"])
        page = (f"<!doctype html><meta charset='utf-8'><title>開催日検証結果</title>"
                f"<h1>{race_date} — {report['status']}</h1>"
                "<p>予想CSV・予想ページには成功したレースだけを掲載しています。失敗・不足は下表を確認してください。</p>"
                f"<p>{html.escape(str(report.get('error', '')))}</p>"
                + ("<p><a href='preview/index.html'>予想ページ</a></p>" if (output / "preview/index.html").exists() else "")
                + table.to_html(index=False, escape=True))
        (output / "index.html").write_text(page, encoding="utf-8")
        summary = (f"## {race_date} 全レース予想テスト: {report['status']}\n\n"
                   f"検出 {len(report['race_ids'])} / 成功 {report['counts']['ok']} / 不足 {report['counts']['incomplete']} / 失敗 {report['counts']['error']}\n\n"
                   "予想は成功分のみ。詳細はArtifactsのindex.html / race_status.csvを確認してください。\n\n"
                   "| レースID | 結果 | 出馬表 | オッズ | DB | 予想 | 照合 | 買い目 |\n|---|---|---|---|---|---|---|---|\n")
        for row in report["races"]:
            summary += "| " + " | ".join(str(row.get(k, "skipped")) for k in ("race_id", "status", "card", "jra_odds", "database", "prediction", "odds_match", "betting")) + " |\n"
        summary += "\n" + str(report.get("error", ""))
        (output / "summary.md").write_text(summary, encoding="utf-8")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(summary)
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    return report


# コマンドライン引数を解析して処理を起動する入口。省略時の保存先やモードはparserで定義する。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-method", choices=["none", "class_weight", "sigmoid"], default="none")
    parser.add_argument("--race-date", type=date.fromisoformat, required=True)
    parser.add_argument("--bundle", type=Path, default=Path("prediction_bundle"))
    parser.add_argument("--output", type=Path, default=Path("data/date_prediction_smoke"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("新しい出力フォルダを指定してください。")
    result = run_date(args.bundle, args.output, args.race_date, calibration_method=args.calibration_method)
    return {"ok": 0, "incomplete": 2, "error": 1}[result["status"]]


# モジュールとしてimportしただけでは実行せず、CLIとして起動された場合だけ入口を呼ぶ。
if __name__ == "__main__":
    raise SystemExit(main())
