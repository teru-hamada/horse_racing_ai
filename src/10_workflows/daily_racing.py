# 日本時間の日付を基準に当日予想と任意の前日結果照合を実行する。
# タスクごとに成功・不足・失敗を記録し、一方が失敗しても他方の成功分を公開可能にする。
# 前日の予想を再計算せず、公開済みCSVを公式結果で精算する。公開準備時には元CSVのハッシュも再確認する。
#
"""Scheduled predictions and previous-day settlement; publish each successful task."""
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
validated_result_status = _import_module('src.50_result_comparison.prediction_comparison').validated_result_status
withdrawals_without_exposure = _import_module('src.50_result_comparison.prediction_comparison').withdrawals_without_exposure
prepare_publication = _import_module('src.60_publication.publish_predictions').prepare_publication
check_meeting_day = _import_module('src.20_scrapers_html_collection.race_calendar').check_meeting_day
build_prediction_site = _import_module('src.60_publication.static_site').build_prediction_site

# 日本標準時（UTC+9）。日付判定と公開ページの時刻表示を統一する。
JST = timezone(timedelta(hours=9))


# UTCの実行時刻を日本時間へ変換してから日付を決める。年末や深夜も同じ規則で扱う。
def japan_today(now=None):
    return (now or datetime.now(timezone.utc)).astimezone(JST).date()


# 前日CSVを固定して結果照合する。再予想せず、公式払戻・返還まで確認できた場合だけokとする。
def compare_previous(day: date, site: Path, output: Path, logger) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    sources = [site / "predictions" / f"{day}.csv", site / "predictions" / f"{day}_bets.csv"]
    if any(not p.is_file() for p in sources):
        raise ValueError("前日の公開済み予想CSVまたは買い目CSVがありません。")
    # 照合開始時点の公開済み予想・買い目CSVの指紋。公開前に同じ入力のままか再確認する。
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
    # 公式結果を取得し、保存済み予想との馬番対応と払戻情報を検査してから照合側へ返す。
    def fetch(race_id, race_date):
        # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
        # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
        actual = fetcher.fetch_result_for_comparison(race_id, race_date, force=True)
        expected = predictions[predictions.race_id.eq(race_id)]
        allowed = withdrawals_without_exposure(actual, expected, bets[bets.race_id.eq(race_id)])
        statuses = validated_result_status(actual, allowed)
        # 公式確定結果に含まれる馬番の集合。
        actual_numbers = set(actual.horse_number)
        # 保存済み予想に含まれる馬番の集合。
        expected_numbers = set(expected.horse_number)
        withdrawn = set(actual.loc[statuses.isin({"scratched", "excluded"}), "horse_number"])
        # Results retain horses already withdrawn before prediction. Only those
        # officially confirmed withdrawals may be absent from published predictions.
        if (actual.horse_number.duplicated().any() or not expected_numbers <= actual_numbers
                or not (actual_numbers - expected_numbers) <= withdrawn):
            raise ValueError("確定結果と予想の馬番が一致しません。")
        payouts = actual.attrs.get("official_payouts", {})
        # 着順だけで精算完了としない。少なくとも単勝の公式払戻があることを確認する。
        if not any(key.startswith("win:") for key in payouts):
            raise ValueError("公式払戻金が不足しています。")
        return actual
    try:
        comparison, summary = compare_prediction_date(predictions, day, fetch, bets=bets)
    finally:
        fetcher.session.close()
    comparison.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    # 推奨買い目へ的中・不的中・返還・未確認と公式払戻額を付けた表。
    judged = compare_recommended_bets(bets, summary["official_payouts"], summary["official_refunds"])
    judged.to_csv(output / "bets_results.csv", index=False, encoding="utf-8-sig")
    # 照合対象と精算結果がすべて揃い、公開可能と判断できるかのフラグ。
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


# 開催判定を含めタスク単位で例外を記録する。開催なしは正常扱いだが取得失敗はerrorになる。
def run_daily(day: date, bundle: Path, site: Path, output: Path, with_previous: bool) -> dict:
    # bundle: runtime.zipとmanifestなどがある配布用バンドルのディレクトリ。
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
            # 処理失敗の原因をレポートへ残す。成功扱いや開催なしへの置き換えは行わない。
            except Exception as exc:
                logger.exception("%s %s の処理失敗", name, target)
                report["tasks"][name] = {"status": "error", "race_date": str(target), "error": str(exc)}
            (output / "daily_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        statuses = [task["status"] for task in report["tasks"].values()]
        report["status"] = "error" if "error" in statuses else (
            "ok" if all(s in {"ok", "no_races"} for s in statuses) else "incomplete")
        report["publication_required"] = "ok" in statuses
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


# 一時サイトで検証・ページ作成を終えてから差分を反映する。照合元CSVの変更は反映前に拒否する。
def prepare_daily_publication(output: Path, site: Path) -> bool:
    report = json.loads((output / "daily_report.json").read_text(encoding="utf-8"))
    # 日次処理がまだ終了していなければ、途中の成果物を公開へ回さない。
    if report["status"] not in {"ok", "incomplete", "error"}:
        raise ValueError("日次処理が終了していないため公開を停止します。")
    # Decide from individual task results, including reports from older runs.
    if not any(t["status"] == "ok" for t in report["tasks"].values()):
        if any(t["status"] != "no_races" for t in report["tasks"].values()):
            raise ValueError("公開できる成功済みの処理がありません。")
        return False
    with tempfile.TemporaryDirectory() as temporary:
        # 公開ファイルを検証・生成する一時ディレクトリ。既存サイトを先に変更しないために使う。
        staged = Path(temporary) / "docs"
        shutil.copytree(site, staged)
        if report["tasks"]["prediction"]["status"] == "ok":
            prepare_publication(output / "prediction", staged)
        task = report["tasks"].get("comparison", {})
        if task.get("status") == "ok":
            root = output / "comparison"
            # 前日照合の状態と元CSVのハッシュ・集計を持つレポート。
            comparison_report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            # 日次集計だけでなく照合成果物自身の成功も確認し、不完全な精算の公開を防ぐ。
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
            # 最新版の必須成果物が欠けた場合は、古いモデルへ自動切替せず不足を通知する。
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


# 日次実行と公開準備を切り替える。通常実行の終了コードは成功0、不足2、失敗1。
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
        # 処理失敗の原因をレポートへ残す。成功扱いや開催なしへの置き換えは行わない。
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


# モジュールとしてimportしただけでは実行せず、CLIとして起動された場合だけ入口を呼ぶ。
if __name__ == "__main__":
    raise SystemExit(main())
