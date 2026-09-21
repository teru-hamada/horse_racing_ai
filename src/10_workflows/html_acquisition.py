"""One-race HTML probe with optional isolated database verification."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import date
from importlib import import_module
from pathlib import Path

import pandas as pd

NetkeibaHtmlCollector = import_module(
    "src.20_scrapers_html_collection.scrapers_html_collection_netkeiba"
).NetkeibaHtmlCollector
JraOddsHtmlCollector = import_module(
    "src.20_scrapers_html_collection.jra_odds_html"
).JraOddsHtmlCollector
parse_odds_directory = import_module(
    "src.30_scrapers_database_creation.jra_odds_parser"
).parse_odds_directory


def inspect_card(frame: pd.DataFrame) -> dict:
    count = len(frame)
    numbers = pd.to_numeric(frame.get("horse_number", pd.Series(dtype=float)), errors="coerce")
    frames = pd.to_numeric(frame.get("frame_number", pd.Series(dtype=float)), errors="coerce")
    odds = pd.to_numeric(frame.get("odds", pd.Series(dtype=float)), errors="coerce")
    valid_numbers = numbers.between(1, 18) & numbers.mod(1).eq(0)
    valid_frames = frames.between(1, 8) & frames.mod(1).eq(0)
    complete = (
        count > 0 and int(valid_numbers.sum()) == count
        and numbers.nunique() == count and int(valid_frames.sum()) == count
    )
    return {
        "status": "ok" if complete else "incomplete",
        "runners": count,
        "horse_numbers": int(valid_numbers.sum()),
        "frame_numbers": int(valid_frames.sum()),
        "card_odds": int(odds.gt(0).sum()),
    }


class DiagnosticOddsCollector(JraOddsHtmlCollector):
    """Keep navigation pages too, so missing meetings can be investigated."""
    def __init__(self, logger, directory: Path):
        super().__init__(logger)
        self.directory = directory
        self.request_count = 0

    def _post(self, cname: str) -> str:
        self.request_count += 1
        html = super()._post(cname)
        self._write_utf8(self.directory / f"response_{self.request_count:02d}.html", html)
        return html


def run_probe(race_date: date, race_id: str, output: Path, *, verify_db: bool = False) -> dict:
    # Require a fresh directory: a failed request must not reuse old HTML.
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f"html_fetch_smoke.{output}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(output / "fetch.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    started = time.monotonic()
    report = {"race_date": race_date.isoformat(), "race_id": race_id}
    frame = None
    odds = None
    try:
        collector = NetkeibaHtmlCollector(logger)
        try:
            # fetch_card also downloads pedigrees; this probe only needs the card.
            html = collector._download(
                collector.CARD_URL.format(race_id=race_id),
                output / "card_raw.html", force=True,
            )
            html = collector._enrich_upcoming_html_with_odds(
                html, race_id, output / "card.html", force=True,
            )
            collector._write_utf8_html(output / "card.html", html)
            frame = collector._parse_page(html, race_id, race_date, "upcoming")
            frame.to_csv(output / "card.csv", index=False, encoding="utf-8-sig")
            report["card"] = inspect_card(frame)
        except Exception as exc:
            logger.exception("出馬表の取得または解析に失敗")
            report["card"] = {"status": "error", "error": str(exc)}
        finally:
            collector.session.close()

        odds_collector = DiagnosticOddsCollector(logger, output / "jra_responses")
        try:
            saved = odds_collector.collect_date(race_date, [race_id], output / "odds", force=True)
            odds = parse_odds_directory(output / "odds", race_date, [race_id])
            odds.to_csv(output / "odds.csv", index=False, encoding="utf-8-sig")
            positive = odds[pd.to_numeric(odds["odds_min"], errors="coerce").gt(0)]
            report["jra_odds"] = {
                "status": "ok" if not positive.empty else "incomplete",
                "html_files": saved, "rows": len(positive),
                "by_bet_type": positive.groupby("bet_type").size().to_dict(),
            }
        except Exception as exc:
            logger.exception("JRAオッズの取得または解析に失敗")
            report["jra_odds"] = {"status": "error", "error": str(exc)}
        finally:
            odds_collector.session.close()

        if verify_db:
            if frame is None or odds is None:
                report["database"] = {"status": "incomplete", "reason": "取得・解析失敗のためDB検証をスキップ"}
            else:
                try:
                    from .database_validation import verify_database
                    report["database"] = verify_database(frame, odds, race_date, race_id, output)
                except Exception as exc:
                    logger.exception("テスト用DBの登録または検証に失敗")
                    report["database"] = {"status": "error", "error": str(exc)}
        sections = ["card", "jra_odds"] + (["database"] if verify_db else [])
        statuses = [report[key]["status"] for key in sections]
        report["status"] = "error" if "error" in statuses else (
            "ok" if all(value == "ok" for value in statuses) else "incomplete"
        )
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = (
            f"## 1レースHTML取得{'・DB登録' if verify_db else ''}テスト: {report['status']}\n\n"
            f"対象日: {race_date} / race_id: {race_id}\n\n"
            f"所要時間: {report['elapsed_seconds']} 秒\n\n"
            "```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n\n"
            "ok: 出馬表の全頭に有効な馬番・枠番があり、JRAオッズを1件以上解析できた。\n\n"
            "incomplete: 必要項目不足。未発売・公開期間外・日付違い・HTML構造変更などを保存HTMLで確認してください。\n\n"
            "全券種・全頭分のオッズ充足や、指定日と出馬表の実開催日の一致までは保証しません。\n"
        )
        if verify_db:
            summary += (
                "\nDB検証: 専用smoke.duckdbに2回登録し、件数・番号・日付・重複・"
                "単勝オッズとの全頭照合を確認します。database.checksはtrueが合格です。\n"
                "取消等で単勝オッズがない馬も要確認としてincompleteになります。\n"
            )
        (output / "summary.md").write_text(summary, encoding="utf-8")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(summary)
        logger.info("検証結果: %s", json.dumps(report, ensure_ascii=False))
        return report
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-date", required=True, type=date.fromisoformat)
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/html_fetch_smoke"))
    parser.add_argument("--verify-db", action="store_true", help="register twice in an isolated smoke.duckdb and validate")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9]{12}", args.race_id):
        parser.error("race-id must be 12 ASCII digits")
    if (args.race_id[:4] != str(args.race_date.year)
            or not 1 <= int(args.race_id[4:6]) <= 10
            or not 1 <= int(args.race_id[6:8]) <= 12
            or not 1 <= int(args.race_id[8:10]) <= 12
            or not 1 <= int(args.race_id[10:]) <= 12):
        parser.error("race-id must match the date year and a valid JRA race")
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    report = run_probe(args.race_date, args.race_id, args.output, verify_db=args.verify_db)
    return {"ok": 0, "incomplete": 2, "error": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
