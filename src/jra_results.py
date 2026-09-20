from __future__ import annotations

import re
import time
import unicodedata
from datetime import date
from importlib import import_module
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PATHS = import_module("src.00_common.config").PATHS


BET_TYPE_BY_CLASS = {
    "win": "win",
    "place": "place",
    "wakuren": "bracket_quinella",
    "umaren": "quinella",
    "wide": "wide",
    "umatan": "exacta",
    "trio": "trio",
    "tierce": "trifecta",
}


class JraResultFetcher:
    """JRA公式のレース結果HTMLを保存し、着順と払戻金を解析する。"""

    URL = "https://www.jra.go.jp/JRADB/accessS.html"
    ENTRY_CNAME = "pw01sli00/AF"
    _ACTION_PATTERN = re.compile(
        r"(?:doAction\(\s*['\"]\/JRADB\/accessS\.html['\"]\s*,\s*['\"]"
        r"|accessS\.html\?CNAME=)([^'\"&]+)",
        re.IGNORECASE,
    )
    _VENUE_PATTERN = re.compile(
        r"^pw01srl\d{2}(?P<course>\d{2})(?P<year>\d{4})"
        r"(?P<meeting>\d{2})(?P<meeting_day>\d{2})(?P<date>\d{8})"
        r"/[0-9A-F]{2}$",
        re.IGNORECASE,
    )
    _RACE_PATTERN = re.compile(
        r"^pw01sde\d{2}(?P<course>\d{2})(?P<year>\d{4})"
        r"(?P<meeting>\d{2})(?P<meeting_day>\d{2})(?P<race>\d{2})"
        r"(?P<date>\d{8})/[0-9A-F]{2}$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        logger,
        interval_seconds: float = 2.0,
        timeout: int = 30,
        session: requests.Session | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.logger = logger
        self.interval_seconds = max(float(interval_seconds), 0.0)
        self.timeout = timeout
        self.session = session or requests.Session()
        self.output_root = output_root or (PATHS.raw_html / "jra" / "historical")
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            ),
            "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
            "Referer": "https://www.jra.go.jp/",
        })
        self._entry_html: str | None = None
        self._race_cnames: dict[str, str] = {}

    @classmethod
    def _action_cnames(cls, html: str) -> list[str]:
        return list(dict.fromkeys(cls._ACTION_PATTERN.findall(html)))

    def _post(self, cname: str) -> str:
        response = self.session.post(
            self.URL,
            data={"cname": cname},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": self.URL,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        response.encoding = "cp932"
        html = response.text
        if "パラメータエラー" in html or "ＤＢ検索エラー" in html:
            raise ValueError("JRA結果ページへのアクセスでエラーが返されました。")
        if self.interval_seconds:
            time.sleep(self.interval_seconds)
        return html

    @staticmethod
    def _write_utf8(path: Path, html: str) -> None:
        normalized = re.sub(
            r'(<meta\b[^>]*\bcharset\s*=\s*["\']?)[^"\'\s/>]+',
            r"\1UTF-8",
            html,
            flags=re.IGNORECASE,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalized, encoding="utf-8", errors="replace")

    @staticmethod
    def _race_parts(race_id: str, race_date: date) -> dict[str, str]:
        if not re.fullmatch(r"\d{12}", race_id):
            raise ValueError(f"JRA結果取得に使用できないrace_idです: {race_id}")
        parts = {
            "year": race_id[0:4],
            "course": race_id[4:6],
            "meeting": race_id[6:8],
            "meeting_day": race_id[8:10],
            "race": race_id[10:12],
            "date": race_date.strftime("%Y%m%d"),
        }
        if parts["year"] != str(race_date.year):
            raise ValueError("race_idの年と比較対象日が一致しません。")
        return parts

    def _discover_race_cname(self, race_id: str, race_date: date) -> str:
        cached = self._race_cnames.get(race_id)
        if cached is not None:
            return cached
        parts = self._race_parts(race_id, race_date)
        if self._entry_html is None:
            self._entry_html = self._post(self.ENTRY_CNAME)

        venue_cname = None
        for cname in self._action_cnames(self._entry_html):
            match = self._VENUE_PATTERN.fullmatch(cname)
            if match and all(match.group(key) == parts[key] for key in (
                "course", "year", "meeting", "meeting_day", "date"
            )):
                venue_cname = cname
                break
        if venue_cname is None:
            raise ValueError(
                f"JRAの開催一覧に対象開催がありません: {race_date} / {race_id}"
            )

        venue_html = self._post(venue_cname)
        for cname in self._action_cnames(venue_html):
            match = self._RACE_PATTERN.fullmatch(cname)
            if match is None:
                continue
            discovered_id = (
                f"{match.group('year')}{match.group('course')}"
                f"{match.group('meeting')}{match.group('meeting_day')}"
                f"{match.group('race')}"
            )
            if match.group("date") == parts["date"]:
                self._race_cnames.setdefault(discovered_id, cname)
        if race_id not in self._race_cnames:
            raise ValueError(f"JRAの開催ページに対象レースがありません: {race_id}")
        return self._race_cnames[race_id]

    def fetch_result_for_comparison(
        self,
        race_id: str,
        race_date: date,
        *,
        force: bool = True,
    ) -> pd.DataFrame:
        """JRA公式結果を保存して解析する。force=Falseなら保存HTMLを再利用する。"""
        race_id = str(race_id)
        cache = self.output_root / str(race_date.year) / "result" / f"{race_id}.html"
        if cache.exists() and not force:
            self.logger.info(f"保存済みJRA結果HTMLを再利用: {cache}")
        else:
            cname = self._discover_race_cname(race_id, race_date)
            html = self._post(cname)
            self._write_utf8(cache, html)
            self.logger.info(f"JRA結果HTML保存: race_id={race_id}, path={cache}")
        frame = parse_jra_result_html(cache, race_id)
        if frame.empty or frame["finish_position"].notna().sum() == 0:
            raise ValueError(
                "JRAから確定着順を取得できませんでした。結果確定後に再実行してください。"
            )
        return frame


def _integer(text: str) -> int | None:
    normalized = unicodedata.normalize("NFKC", text).replace(",", "")
    match = re.search(r"\d+", normalized)
    return int(match.group()) if match else None


def _selection(text: str, *, unordered: bool) -> str:
    values = [int(value) for value in re.findall(
        r"\d+", unicodedata.normalize("NFKC", text)
    )]
    if unordered:
        values.sort()
    return "-".join(map(str, values))


def parse_jra_result_html(path: Path, race_id: str) -> pd.DataFrame:
    """保存済みJRA結果HTMLから着順と100円当たりの公式払戻金を読む。"""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    result_table = next(
        (table for table in soup.find_all("table") if table.select_one("th.place")),
        None,
    )
    rows: list[dict[str, object]] = []
    if result_table is not None:
        for tr in result_table.select("tbody tr"):
            horse_number = _integer(
                tr.select_one("td.num").get_text(" ", strip=True)
            ) if tr.select_one("td.num") else None
            if horse_number is None:
                continue
            frame_node = tr.select_one("td.waku")
            frame_number = _integer(frame_node.get_text(" ", strip=True)) if frame_node else None
            if frame_number is None and frame_node and frame_node.find("img"):
                frame_number = _integer(frame_node.find("img").get("alt", ""))
            rows.append({
                "race_id": race_id,
                "horse_id": pd.NA,
                "horse_number": horse_number,
                "horse_name": tr.select_one("td.horse").get_text(" ", strip=True)
                if tr.select_one("td.horse") else pd.NA,
                "frame_number": frame_number,
                "finish_position": _integer(
                    tr.select_one("td.place").get_text(" ", strip=True)
                ) if tr.select_one("td.place") else None,
            })

    payouts: dict[str, int] = {}
    unordered_types = {"bracket_quinella", "quinella", "wide", "trio"}
    for class_name, bet_type in BET_TYPE_BY_CLASS.items():
        for line in soup.select(f".refund_area li.{class_name} .line"):
            num = line.select_one(".num")
            yen = line.select_one(".yen")
            if num is None or yen is None:
                continue
            selection = _selection(
                num.get_text(" ", strip=True), unordered=bet_type in unordered_types
            )
            payout = _integer(yen.get_text(" ", strip=True))
            if selection and payout is not None:
                payouts[f"{bet_type}:{selection}"] = payout

    frame = pd.DataFrame(rows, columns=[
        "race_id", "horse_id", "horse_number", "horse_name",
        "frame_number", "finish_position",
    ])
    frame.attrs["official_payouts"] = payouts
    frame.attrs["source_html"] = str(path)
    return frame
