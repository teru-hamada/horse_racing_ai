"""Scheduled predictions and previous-day settlement; publish only complete runs."""
from __future__ import annotations
from importlib import import_module as _import_module

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd

from .date_prediction import run_date

DateCollector = _import_module('src.20_scrapers_html_collection.date_collector').DateCollector
JraResultFetcher = _import_module('src.20_scrapers_html_collection.jra_results').JraResultFetcher
sha256 = _import_module('src.40_ai_modeling.prediction_bundle').sha256
compare_prediction_date = _import_module('src.50_result_comparison.prediction_comparison').compare_prediction_date
compare_recommended_bets = _import_module('src.50_result_comparison.prediction_comparison').compare_recommended_bets
prepare_publication = _import_module('src.60_publication.publish_predictions').prepare_publication
check_meeting_day = _import_module('src.20_scrapers_html_collection.race_calendar').check_meeting_day
build_prediction_site = _import_module('src.60_publication.static_site').build_prediction_site

JST = timezone(timedelta(hours=9))


def japan_today(now=None):
    return (now or datetime.now(timezone.utc)).astimezone(JST).date()


def compare_previous(day: date, site: Path, output: Path, logger) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    sources = [site / "predictions" / f"{day}.csv", site / "predictions" / f"{day}_bets.csv"]
    if any(not p.is_file() for p in sources):
        raise ValueError("前日の公開済み予想CSVまたは買い目CSVがありません。")
    source_hashes = {p.name: sha256(p) for p in sources}
    for source in sources:
        shutil.copy2(source, output / source.name)
    predictions = pd.read_csv(output / sources[0].name, dtype={"race_id": str, "horse_id": str})
    bets = pd.read_csv(output / sources[1].name, dtype={"race_id": str, "selection": str})
    if predictions.empty or not pd.to_datetime(predictions.race_date).dt.date.eq(day).all():
        raise ValueError("前日予想の開催日またはデータが不正です。")
    collector = DateCollector(logger, output / "discovery")
    try:
        ids = set(collector.race_ids_for_date(day, str(day.year), "upcoming", force=True))
    finally:
        collector.session.close()
    if not ids or ids != set(predictions.race_id):
        raise ValueError("前日開催一覧と公開済み予想のレースが一致しません。")
    fetcher = JraResultFetcher(logger, output_root=output / "jra")
    def fetch(race_id, race_date):
        actual = fetcher.fetch_result_for_comparison(race_id, race_date, force=True)
        expected = predictions[predictions.race_id.eq(race_id)]
        if actual.horse_number.duplicated().any() or set(actual.horse_number) != set(expected.horse_number):
            raise ValueError("確定結果と予想の馬番が一致しません。")
        payouts = actual.attrs.get("official_payouts", {})
        if not any(key.startswith("win:") for key in payouts):
            raise ValueError("公式払戻金が不足しています。")
        return actual
    try:
        comparison, summary = compare_prediction_date(predictions, day, fetch)
    finally:
        fetcher.session.close()
    comparison.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    judged = compare_recommended_bets(bets, summary["official_payouts"], summary["official_refunds"])
    judged.to_csv(output / "bets_results.csv", index=False, encoding="utf-8-sig")
    complete = summary["failed_races"] == 0 and summary["compared_races"] == len(ids)
    complete = complete and not judged.bet_result.eq("未確認").any()
    for failure in summary["failures"]:
        logger.warning("結果照合失敗: race_id=%s, %s", failure["race_id"], failure["message"])
    if judged.bet_result.eq("未確認").any():
        logger.warning("払戻・返還を確認できない買い目が %s 点あります。", int(judged.bet_result.eq("未確認").sum()))
    result = {"status": "ok" if complete else "incomplete", "race_date": str(day),
              "source_hashes": source_hashes, "summary": summary}
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_daily(day: date, bundle: Path, site: Path, output: Path, with_previous: bool) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"daily.{output}")
    logger.setLevel(logging.INFO)
    handlers = [logging.StreamHandler(), logging.FileHandler(output / "daily.log", encoding="utf-8")]
    for handler in handlers:
        logger.addHandler(handler)
    report = {"race_date": str(day), "status": "running", "tasks": {}}
    tasks = [("prediction", day)] + ([("comparison", day - timedelta(days=1))] if with_previous else [])
    try:
        for name, target in tasks:
            try:
                meeting = check_meeting_day(target, output / "calendar" / name, logger)
                if not meeting:
                    result = {"status": "no_races", "race_date": str(target)}
                elif name == "prediction":
                    result = run_date(bundle, output / "prediction", target)
                else:
                    result = compare_previous(target, site, output / "comparison", logger)
                report["tasks"][name] = {"status": result["status"], "race_date": str(target)}
                if result.get("error"):
                    report["tasks"][name]["error"] = result["error"]
            except Exception as exc:
                logger.exception("%s %s の処理失敗", name, target)
                report["tasks"][name] = {"status": "error", "race_date": str(target), "error": str(exc)}
            (output / "daily_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        statuses = [task["status"] for task in report["tasks"].values()]
        report["status"] = "error" if "error" in statuses else (
            "ok" if all(s in {"ok", "no_races"} for s in statuses) else "incomplete")
        report["publication_required"] = report["status"] == "ok" and "ok" in statuses
    finally:
        text = json.dumps(report, ensure_ascii=False, indent=2)
        (output / "daily_report.json").write_text(text, encoding="utf-8")
        summary = "## 日次予想・結果照合\n\n```json\n" + text + "\n```\n"
        (output / "summary.md").write_text(summary, encoding="utf-8")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(summary)
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    return report


def prepare_daily_publication(output: Path, site: Path) -> bool:
    report = json.loads((output / "daily_report.json").read_text(encoding="utf-8"))
    if report["status"] != "ok" or any(t["status"] not in {"ok", "no_races"} for t in report["tasks"].values()):
        raise ValueError("日次処理が未合格のため公開を停止します。")
    if not report["publication_required"]:
        return False
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / "docs"
        shutil.copytree(site, staged)
        if report["tasks"]["prediction"]["status"] == "ok":
            prepare_publication(output / "prediction", staged)
        task = report["tasks"].get("comparison", {})
        if task.get("status") == "ok":
            root = output / "comparison"
            comparison_report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            if comparison_report["status"] != "ok":
                raise ValueError("前日照合が未合格です。")
            day = date.fromisoformat(task["race_date"]).isoformat()
            if set(comparison_report["source_hashes"]) != {f"{day}.csv", f"{day}_bets.csv"}:
                raise ValueError("照合元のハッシュ情報が不足しています。")
            for name, digest in comparison_report["source_hashes"].items():
                if name not in {f"{day}.csv", f"{day}_bets.csv"} or sha256(site / "predictions" / name) != digest:
                    raise ValueError("前日予想が照合開始後に変更されました。再実行してください。")
            predictions = pd.read_csv(root / f"{day}.csv", dtype={"race_id": str, "horse_id": str})
            bets = pd.read_csv(root / f"{day}_bets.csv", dtype={"race_id": str, "selection": str})
            comparison = pd.read_csv(root / "comparison.csv", dtype={"race_id": str, "horse_id": str})
            build_prediction_site(predictions, day, predictions.model_run_id.iloc[0], staged,
                                  comparison=comparison, comparison_summary=comparison_report["summary"], bet_recommendations=bets)
            for source, target in (("comparison.csv", f"{day}_comparison.csv"), ("bets_results.csv", f"{day}_bets_results.csv")):
                shutil.copy2(root / source, staged / "predictions" / target)
        pattern = '<p class="meta">生成日時: [^<]*</p>'.encode("utf-8")
        for source in staged.rglob("*"):
            if not source.is_file():
                continue
            target = site / source.relative_to(staged)
            content = source.read_bytes()
            if target.exists():
                previous = target.read_bytes()
                if source.suffix == ".html":
                    if re.sub(pattern, b'', previous.replace(b'\r\n', b'\n')) == re.sub(pattern, b'', content.replace(b'\r\n', b'\n')):
                        continue
                elif previous == content:
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-date", type=date.fromisoformat)
    parser.add_argument("--with-previous", action="store_true")
    parser.add_argument("--prepare-publication", action="store_true")
    parser.add_argument("--bundle", type=Path, default=Path("prediction_bundle"))
    parser.add_argument("--site", type=Path, default=Path("docs"))
    parser.add_argument("--output", type=Path, default=Path("data/daily_racing"))
    args = parser.parse_args()
    if args.prepare_publication:
        try:
            required = prepare_daily_publication(args.output, args.site)
            publication = {"status": "ok", "publication_required": required}
        except Exception as exc:
            publication = {"status": "error", "error": str(exc)}
            (args.output / "publication_report.json").write_text(json.dumps(publication, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        (args.output / "publication_report.json").write_text(json.dumps(publication, ensure_ascii=False, indent=2), encoding="utf-8")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
                stream.write(f"publication_required={str(required).lower()}\n")
        return 0
    if args.output.exists():
        parser.error("出力先は新しいフォルダを指定してください。")
    report = run_daily(args.race_date or japan_today(), args.bundle, args.site, args.output, args.with_previous)
    return {"ok": 0, "incomplete": 2, "error": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
