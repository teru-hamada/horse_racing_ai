# JRAの公開画面をたどり、対象日のレース・券種別オッズHTMLを保存する。
# POST遷移用の識別子からレースIDと券種を読み取り、対象外の画面を除く。
# 保存時にはHTMLの文字コード宣言をUTF-8へ統一し、解析処理は別モジュールへ委ねる。
#
from __future__ import annotations

import re
import json
import hashlib
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


# JRAのPOST画面遷移を管理し、レース・券種ごとのHTMLキャッシュを収集する。
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

    # 取得先のルート、HTTPセッション、ログ、通信条件をインスタンスへ保持する。
    def __init__(
        self,
        logger,
        interval_seconds: float = 2.0,
        timeout: int = 30,
        session: requests.Session | None = None,
    ) -> None:
        # 取得や解析の経過・警告・例外を通知するロガー。
        self.logger = logger
        # 取得先への連続アクセスを避けるための要求間隔（秒）。
        self.interval_seconds = max(float(interval_seconds), 0.0)
        # 一回のHTTP要求に対する待機上限秒数。
        self.timeout = timeout
        # 再利用するHTTPセッション。テストでは偽セッションへ差し替える。
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

    # JRAページのリンクやイベント属性から、POST遷移に使う画面識別子を抽出する。
    @classmethod
    def _action_cnames(cls, html: str) -> list[str]:
        return list(dict.fromkeys(cls._ACTION_PATTERN.findall(html)))

    # JRAの画面識別子をレースIDと券種へ解釈する。対象形式でなければ採用しない。
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

    # JRAの画面遷移パラメーターをPOSTし、診断に使える画面本文を受け取る。
    def _post(self, cname: str) -> str:
        for attempt in range(4):
            try:
                return self._post_once(cname)
            except requests.exceptions.SSLError:
                raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt == 3:
                    raise
                delay = max(self.interval_seconds, 2.0 * (2 ** attempt))
                self.logger.warning(
                    f"JRAオッズ通信失敗: cname={cname}, "
                    f"retry={attempt + 1}/3, wait={delay:g}s: {exc}"
                )
                time.sleep(delay)

    def _post_once(self, cname: str) -> str:
        response = self.session.post(
            self.URL,
            data={"cname": cname},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": self.URL,
            },
            timeout=self.timeout,
        )
        self._last_fetched_at = datetime.now(timezone.utc).isoformat()
        response.raise_for_status()
        response.encoding = "cp932"
        html = response.text
        if self.interval_seconds:
            time.sleep(self.interval_seconds)
        return html

    # 保存前に文字コード宣言を統一し、ローカル解析でも表示でも同じ文字列を扱えるようにする。
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

    # 対象日・レースIDへ絞って券種ごとのHTMLを保存し、取得件数と進捗を呼出元へ返す。
    def collect_date(
        self,
        target_date: date,
        race_ids: Iterable[str],
        output_directory: Path,
        force: bool = False,
        progress_callback=None,
    ) -> int:
        """取得可能な対象レース・券種のHTMLを一件ずつ保存する。"""
        # force: キャッシュを使わず新規取得するかを指定する。
        target_race_ids = set(race_ids)
        if not target_race_ids:
            # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
            if progress_callback:
                progress_callback(1.0)
            return 0

        entry_html = self._post(self.ENTRY_CNAME)
        # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
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
            # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
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
            # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
            if progress_callback:
                progress_callback(
                    0.1 + 0.2 * venue_index / len(venue_cnames)
                )

        saved_count = 0
        sorted_pages = sorted(pages.items())
        if not sorted_pages:
            # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
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
                    # キャッシュ再利用時は更新せず、実際のHTTP取得時刻を保持する。
                    metadata = {"fetched_at": getattr(self, "_last_fetched_at", None),
                                "source_url": self.URL, "source_cname": cname,
                                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    path.with_suffix(".metadata.json").write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                    saved_count += 1
                    self.logger.info(
                        f"JRAオッズHTML保存: race_id={race_id}, "
                        f"bet_type={bet_type}, path={path}"
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        f"{race_id} {bet_type}: JRAオッズHTML取得失敗: {exc}"
                    )
            # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
            if progress_callback:
                progress_callback(
                    0.3 + 0.7 * page_index / len(sorted_pages)
                )
        return saved_count
