from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import requests


class JraOddsHtmlCollector:
    """JRAの公開オッズ画面をたどり、券種別HTMLを保存する。"""

    URL = "https://www.jra.go.jp/JRADB/accessO.html"
    ENTRY_CNAME = "pw15oli00/6D"
    BET_TYPES = {
        "1": "win_place",
        "3": "bracket_quinella",
        "4": "quinella",
        "5": "wide",
        "6": "exacta",
        "7": "trio",
        "8": "trifecta",
    }
    _ACTION_PATTERN = re.compile(
        r"doAction\(\s*['\"]\/JRADB\/accessO\.html(?:#[^'\"]*)?['\"]\s*,"
        r"\s*['\"]([^'\"]+)['\"]"
    )
    _RACE_CNAME_PATTERN = re.compile(
        r"^pw15(?P<bet>[1-8])ouS3"
        r"(?P<course>\d{2})(?P<year>\d{4})(?P<meeting>\d{2})"
        r"(?P<meeting_day>\d{2})(?P<race>\d{2})(?P<date>\d{8})"
        r"Z(?:99)?/[0-9A-F]{2}$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        logger,
        interval_seconds: float = 2.0,
        timeout: int = 30,
        session: requests.Session | None = None,
    ) -> None:
        self.logger = logger
        self.interval_seconds = max(float(interval_seconds), 0.0)
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                ),
                "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
                "Referer": "https://www.jra.go.jp/",
            }
        )

    @classmethod
    def _action_cnames(cls, html: str) -> list[str]:
        return list(dict.fromkeys(cls._ACTION_PATTERN.findall(html)))

    @classmethod
    def _race_id_and_bet_type(
        cls, cname: str, target_date: date
    ) -> tuple[str, str] | None:
        match = cls._RACE_CNAME_PATTERN.fullmatch(cname)
        if match is None or match.group("date") != target_date.strftime("%Y%m%d"):
            return None
        bet_type = cls.BET_TYPES.get(match.group("bet"))
        if bet_type is None:
            return None
        race_id = (
            f"{match.group('year')}{match.group('course')}"
            f"{match.group('meeting')}{match.group('meeting_day')}"
            f"{match.group('race')}"
        )
        return race_id, bet_type

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
        if self.interval_seconds:
            time.sleep(self.interval_seconds)
        return html

    @staticmethod
    def _write_utf8(path: Path, html: str) -> None:
        html = re.sub(
            r'(<meta\b[^>]*\bcharset\s*=\s*["\']?)[^"\'\s/>]+',
            r"\1UTF-8",
            html,
            flags=re.IGNORECASE,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8", errors="replace")

    def collect_date(
        self,
        target_date: date,
        race_ids: Iterable[str],
        output_directory: Path,
        force: bool = False,
        progress_callback=None,
    ) -> int:
        """取得可能な対象レース・券種のHTMLを一件ずつ保存する。"""
        target_race_ids = set(race_ids)
        if not target_race_ids:
            if progress_callback:
                progress_callback(1.0)
            return 0

        entry_html = self._post(self.ENTRY_CNAME)
        if progress_callback:
            progress_callback(0.1)
        date_text = target_date.strftime("%Y%m%d")
        venue_cnames = [
            cname
            for cname in self._action_cnames(entry_html)
            if cname.startswith("pw15orl") and date_text in cname
        ]
        if not venue_cnames:
            self.logger.warning(
                f"{target_date}: JRAオッズページに対象開催がありません。"
            )
            if progress_callback:
                progress_callback(1.0)
            return 0

        pages: dict[tuple[str, str], str] = {}
        for venue_index, venue_cname in enumerate(venue_cnames, start=1):
            venue_html = self._post(venue_cname)
            for cname in self._action_cnames(venue_html):
                parsed = self._race_id_and_bet_type(cname, target_date)
                if parsed is None or parsed[0] not in target_race_ids:
                    continue
                pages.setdefault(parsed, cname)
            if progress_callback:
                progress_callback(
                    0.1 + 0.2 * venue_index / len(venue_cnames)
                )

        saved_count = 0
        sorted_pages = sorted(pages.items())
        if not sorted_pages:
            if progress_callback:
                progress_callback(1.0)
            return 0
        for page_index, ((race_id, bet_type), cname) in enumerate(
            sorted_pages, start=1
        ):
            path = output_directory / f"{race_id}_{bet_type}.html"
            if path.exists() and not force:
                self.logger.info(f"JRAオッズHTMLを再利用: {path}")
                saved_count += 1
            else:
                try:
                    html = self._post(cname)
                    self._write_utf8(path, html)
                    saved_count += 1
                    self.logger.info(
                        f"JRAオッズHTML保存: race_id={race_id}, "
                        f"bet_type={bet_type}, path={path}"
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        f"{race_id} {bet_type}: JRAオッズHTML取得失敗: {exc}"
                    )
            if progress_callback:
                progress_callback(
                    0.3 + 0.7 * page_index / len(sorted_pages)
                )
        return saved_count
