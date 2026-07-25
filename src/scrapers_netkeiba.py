from __future__ import annotations

import re
import shutil
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import PATHS
from .logging_utils import AppLogger


COURSES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

JRA_COURSE_CODE_MAP = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

# JRA中央競馬の競馬場コード
# 01:札幌 02:函館 03:福島 04:新潟 05:東京
# 06:中山 07:中京 08:京都 09:阪神 10:小倉
JRA_COURSE_CODES = {f"{number:02d}" for number in range(1, 11)}
PAGE_REQUEST_INTERVAL_SECONDS = 2.0


def _flatten_columns(columns: pd.Index) -> list[str]:
    """
    pandasで取得した表の列名を平坦化し、列名中の空白を除去する。

    例:
        "着 順" -> "着順"
        "枠 番" -> "枠番"
        "馬 番" -> "馬番"
        "人 気" -> "人気"
    """
    flattened: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [
                str(x)
                for x in col
                if str(x) != "nan"
                and not str(x).startswith("Unnamed")
            ]
            name = "".join(parts)
        else:
            name = str(col)

        normalized = re.sub(r"\s+", "", name)
        flattened.append(normalized)

    return flattened


def _first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        for col in df.columns:
            if candidate in str(col):
                return str(col)
    return None


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _finish_position(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    match = re.match(r"(\d+)", text)
    return float(match.group(1)) if match else None


def _time_to_seconds(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not re.match(r"^\d+:\d+(?:\.\d+)?$", text):
        return None
    minute, second = text.split(":", 1)
    return float(minute) * 60 + float(second)


def _parse_sex_age(value: object) -> tuple[str | None, int | None]:
    text = "" if value is None else str(value).strip()
    match = re.search(r"([牡牝セ騙])\s*(\d+)", text)
    if not match:
        return None, None
    sex = "セ" if match.group(1) == "騙" else match.group(1)
    return sex, int(match.group(2))


def _parse_body_weight(value: object) -> tuple[float | None, float | None]:
    text = "" if value is None else str(value).strip()
    match = re.search(r"(\d+)\s*\(([+-]?\d+)\)", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    plain = re.search(r"(\d+)", text)
    return (float(plain.group(1)), None) if plain else (None, None)


def _safe_filename_part(value: object, max_length: int = 80) -> str:
    """Windowsで使用できない文字を除去してファイル名用文字列を作る。"""
    text = "" if value is None else str(value).strip()
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text).strip(" ._")
    return text[:max_length] or "レース名不明"


class NetkeibaScraper:
    """Small, cache-friendly scraper for netkeiba race-list, card and result pages.

    The parser intentionally uses multiple fallbacks and saves every HTML page. If the source
    markup changes, the raw page remains available for parser adjustment without re-downloading.
    """

    RACE_LIST_URL = "https://race.netkeiba.com/top/race_list.html?kaisai_date={date}"
    RACE_DATE_LIST_URL = (
        "https://race.netkeiba.com/top/"
        "race_list_get_date_list.html"
        "?kaisai_date={date}&encoding=UTF-8"
    )
    RACE_LIST_SUB_URL = (
        "https://race.netkeiba.com/top/race_list_sub.html"
        "?kaisai_date={date}&current_group={current_group}"
    )
    RACE_LIST_FALLBACK_URL = "https://db.netkeiba.com/race/list/{date}/"
    CARD_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    # 過去結果では使用しない。将来の拡張用に定義のみ残す。
    RESULT_URL = "https://race.netkeiba.com/race/result.html?race_id={race_id}"
    RESULT_FALLBACK_URL = "https://db.netkeiba.com/race/{race_id}/"
    HORSE_PEDIGREE_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"

    def __init__(self, logger: AppLogger, interval_seconds: float = 2.0, timeout: int = 30) -> None:
        self.logger = logger
        self.interval_seconds = max(interval_seconds, 0.5)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                ),
                "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
                "Referer": "https://race.netkeiba.com/",
            }
        )
        # app.py側で取得状況を表示できるよう、アクセス情報を保持する。
        self.current_url: str | None = None
        self.last_url: str | None = None
        self.requested_urls: list[str] = []
        self.race_urls: list[str] = []
        self.diagnostics: list[dict[str, object]] = []
        self.pedigree_cache: dict[
            tuple[str, str],
            dict[str, str | None],
        ] = {}

    @staticmethod
    def _html_root(dataset_type: str) -> Path:
        if dataset_type == "historical":
            return PATHS.historical_html
        if dataset_type == "upcoming":
            return PATHS.upcoming_html
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    def _cache_path(
        self,
        dataset_type: str,
        storage_key: str,
        kind: str,
        key: str,
    ) -> Path:
        """年単位の保存先を返す（storage_key は YYYY）。"""
        if kind == "horse":
            directory = self._html_root(dataset_type) / "horse"
        else:
            directory = self._html_root(dataset_type) / storage_key / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{key}.html"

    @staticmethod
    def _normalize_html_charset(html: str) -> str:
        """
        UTF-8で保存するHTML内の文字コード指定もUTF-8へ統一する。
        """
        html = re.sub(
            r'(<meta\b[^>]*\bcharset\s*=\s*["\']?)'
            r'[^"\'\s/>]+',
            r'\1UTF-8',
            html,
            flags=re.IGNORECASE,
        )
        html = re.sub(
            r'(content\s*=\s*["\'][^"\']*?charset\s*=\s*)'
            r'[^;"\'\s/>]+',
            r'\1UTF-8',
            html,
            flags=re.IGNORECASE,
        )
        return html

    def _write_utf8_html(
        self,
        cache_path: Path,
        html: str,
    ) -> str:
        normalized = self._normalize_html_charset(html)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            normalized,
            encoding="utf-8",
            errors="replace",
        )
        return normalized

    def _find_existing_html(
        self,
        cache_path: Path,
    ) -> Path | None:
        """
        別の年フォルダで保存済みの同種HTMLを検索する。

        例:
            raw_html/historical/2026/result/202602011003.html
            raw_html/upcoming/2026/race_list_db/20260712.html
        """
        kind = cache_path.parent.name
        filename = cache_path.name
        key = cache_path.stem

        if kind == "horse":
            return None

        # result/cardは先頭12桁のrace_idをキーに検索する。
        # 例: 202610020612.html と
        #     202610020612_2026年10月02日_レース名.html の双方を対象にする。
        if kind in {"result", "card"} and re.fullmatch(r"\d{12}", key):
            filename_pattern = f"{key}*.html"
        else:
            filename_pattern = filename

        candidates = [
            path
            for path in cache_path.parents[2].glob(
                f"*/{kind}/{filename_pattern}"
            )
            if path.is_file()
            and path.resolve() != cache_path.resolve()
        ]

        if not candidates:
            return None

        # 更新日時が最も新しい保存済みHTMLを利用する。
        return max(
            candidates,
            key=lambda path: path.stat().st_mtime,
        )

    def _download(self, url: str, cache_path: Path, force: bool = False) -> str:
        self.current_url = url
        self.last_url = url
        if url not in self.requested_urls:
            self.requested_urls.append(url)

        if not cache_path.exists() and not force:
            current_kind = cache_path.parent.name
            current_key = cache_path.stem
            if (
                (
                    current_kind in {"result", "card"}
                    and re.fullmatch(r"\d{12}", current_key)
                )
                or (
                    current_kind == "horse"
                    and re.fullmatch(
                        r"[0-9a-zA-Z]+",
                        current_key,
                    )
                )
            ):
                current_candidates = sorted(
                    cache_path.parent.glob(
                        f"{current_key}*.html"
                    ),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if current_candidates:
                    cache_path = current_candidates[0]

        if cache_path.exists() and not force:
            self.logger.info(
                f"現在runのキャッシュを使用: {cache_path} / URL: {url}"
            )
            html = cache_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            self.diagnostics.append(
                {
                    "url": url,
                    "cache": True,
                    "cache_source": str(cache_path),
                    "html_length": len(html),
                    "race_id_count": len(
                        set(
                            re.findall(
                                r"(?<!\d)\d{12}(?!\d)",
                                html,
                            )
                        )
                    ),
                }
            )
            return html

        if not force:
            existing_cache = self._find_existing_html(cache_path)
            if existing_cache is not None:
                cache_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                shutil.copy2(existing_cache, cache_path)

                self.logger.info(
                    "保存済みHTMLを再利用: "
                    f"{existing_cache} -> {cache_path} / URL: {url}"
                )

                html = cache_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                html = self._write_utf8_html(cache_path, html)
                self.diagnostics.append(
                    {
                        "url": url,
                        "cache": True,
                        "cache_source": str(existing_cache),
                        "cache_copy": str(cache_path),
                        "html_length": len(html),
                        "race_id_count": len(
                            set(
                                re.findall(
                                    r"(?<!\d)\d{12}(?!\d)",
                                    html,
                                )
                            )
                        ),
                    }
                )
                return html

        self.logger.info(f"取得開始: {url}")
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        # netkeibaの旧DBページはEUC-JP系の場合があるため、
        # requestsの推定結果を優先しつつ安全にデコードする。
        encoding = response.apparent_encoding or response.encoding or "utf-8"
        response.encoding = encoding
        html = response.text

        html = self._write_utf8_html(cache_path, html)
        self.logger.info(
            f"取得完了: status={response.status_code}, encoding={encoding}, "
            f"html_length={len(html)}, URL={url}"
        )
        self.diagnostics.append(
            {
                "url": url,
                "cache": False,
                "status_code": response.status_code,
                "encoding": encoding,
                "html_length": len(html),
                "race_id_count": len(set(re.findall(r"(?<!\d)\d{12}(?!\d)", html))),
            }
        )
        time.sleep(self.interval_seconds)
        return html

    @staticmethod
    def _extract_race_ids(html: str, date_text: str) -> list[str]:
        """現行・旧DB双方のHTMLから対象日の12桁レースIDを抽出する。"""
        patterns = [
            r"[?&]race_id=(\d{12})(?:&|[\"'<>\s]|$)",
            r"/race/(\d{12})/?(?:[?\"'<>\s]|$)",
            r"race_id[=/](\d{12})",
            r"(?<!\d)(\d{12})(?!\d)",
        ]
        found: set[str] = set()
        for pattern in patterns:
            found.update(re.findall(pattern, html))

        # race_idの先頭4桁は年。別日・広告等のID混入を抑制する。
        year_prefix = date_text[:4]
        return sorted(
            race_id
            for race_id in found
            if race_id.startswith(year_prefix)
            and race_id.isdigit()
            and len(race_id) == 12
            # race_idの5～6桁目が競馬場コード。
            # 01～10だけを許可し、地方競馬を除外する。
            and race_id[4:6] in JRA_COURSE_CODES
            and race_id[-2:] in {f"{number:02d}" for number in range(1, 13)}
        )

    @staticmethod
    def _extract_current_group(
        html: str,
        date_text: str,
    ) -> str | None:
        soup = BeautifulSoup(html, "lxml")
        for node in soup.select("[date][group]"):
            if str(node.get("date", "")) == date_text:
                current_group = str(
                    node.get("group", "")
                ).strip()
                if current_group:
                    return current_group
        return None

    def race_ids_for_date(
        self,
        target_date: date,
        run_id: str,
        dataset_type: str,
        force: bool = False,
    ) -> list[str]:
        """対象日の中央競馬レースIDを取得する。"""
        date_text = target_date.strftime("%Y%m%d")

        if dataset_type == "upcoming":
            date_list_url = self.RACE_DATE_LIST_URL.format(
                date=date_text
            )
            self.logger.info(
                f"開催日一覧取得対象URL: {date_list_url}"
            )
            date_list_html = self._download(
                date_list_url,
                self._cache_path(
                    dataset_type,
                    run_id,
                    "race_date_list",
                    date_text,
                ),
                force=force,
            )
            current_group = self._extract_current_group(
                date_list_html,
                date_text,
            )
            if not current_group:
                raise ValueError(
                    f"{target_date}: current_groupを"
                    "取得できませんでした。"
                )
            list_url = self.RACE_LIST_SUB_URL.format(
                date=date_text,
                current_group=current_group,
            )
        else:
            list_url = self.RACE_LIST_FALLBACK_URL.format(
                date=date_text
            )

        self.logger.info(f"レース一覧取得対象URL: {list_url}")
        list_cache_path = self._cache_path(
            dataset_type,
            run_id,
            "race_list_db",
            date_text,
        )
        html = self._download(
            list_url,
            list_cache_path,
            force=force,
        )

        race_ids = self._extract_race_ids(html, date_text)
        if (
            dataset_type == "upcoming"
            and not force
            and not race_ids
        ):
            self.logger.warning(
                f"{target_date}: 保存済みレース一覧に"
                "race_idがないため、現行一覧を再取得します。"
            )
            html = self._download(
                list_url,
                list_cache_path,
                force=True,
            )
            race_ids = self._extract_race_ids(
                html,
                date_text,
            )
        self.logger.info(
            f"{target_date}: race_list_dbから中央競馬{len(race_ids)}レースを検出 / "
            f"URL: {list_url}"
        )

        self.race_urls = [
            self.RESULT_URL.format(race_id=race_id)
            for race_id in race_ids
        ]
        for url in self.race_urls:
            self.logger.info(f"取得予定URL: {url}")

        if not race_ids:
            self.logger.error(
                f"{target_date}: レースIDを取得できませんでした。"
                "raw_htmlのrace_list_dbを確認してください。"
            )
        return race_ids

    def collect_date_range(
        self,
        start_date: date,
        end_date: date,
        dataset_type: str,
        run_id: str,
        force: bool = False,
        progress_callback=None,
    ) -> pd.DataFrame:
        if end_date < start_date:
            raise ValueError("終了日は開始日以降にしてください。")
        dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        all_records: list[pd.DataFrame] = []
        total_dates = len(dates)
        for date_index, target_date in enumerate(dates, start=1):
            storage_key = str(target_date.year)
            race_ids = self.race_ids_for_date(
                target_date,
                storage_key,
                dataset_type,
                force=force,
            )
            if not race_ids:
                self.logger.error(
                    f"{target_date}: 詳細ページへ進めるレースIDが0件です。"
                )
                if progress_callback:
                    progress_callback(date_index / total_dates)
                continue

            for race_index, race_id in enumerate(race_ids, start=1):
                try:
                    if dataset_type == "historical":
                        frame = self.fetch_result(
                            race_id, target_date, storage_key, force=force
                        )
                    else:
                        frame = self.fetch_card(
                            race_id, target_date, storage_key, force=force
                        )
                    if not frame.empty:
                        all_records.append(frame)
                    self.logger.info(f"{target_date} {race_id}: {len(frame)}頭を解析")
                except Exception as exc:  # noqa: BLE001 - continue other races and retain raw HTML/logs
                    self.logger.error(f"{target_date} {race_id}: 取得・解析失敗: {exc}")
                if progress_callback:
                    fraction = ((date_index - 1) + race_index / max(len(race_ids), 1)) / total_dates
                    progress_callback(min(float(fraction), 1.0))
        return pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()

    def collect_html_date_range(
        self,
        start_date: date,
        end_date: date,
        dataset_type: str,
        force: bool = False,
        progress_callback=None,
    ) -> dict[str, int]:
        """HTMLだけを取得する。解析やDB保存は行わない。"""
        if end_date < start_date:
            raise ValueError("終了日は開始日以降にしてください。")
        if dataset_type not in {"historical", "upcoming"}:
            raise ValueError(f"未対応のdataset_typeです: {dataset_type}")

        dates = [
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        ]
        list_count = 0
        race_count = 0
        collected_horse_ids: set[str] = set()
        total_dates = len(dates)
        for date_index, target_date in enumerate(dates, start=1):
            storage_key = str(target_date.year)
            race_ids = self.race_ids_for_date(
                target_date,
                storage_key,
                dataset_type,
                force=force,
            )
            list_count += 1
            for race_index, race_id in enumerate(race_ids, start=1):
                try:
                    if dataset_type == "historical":
                        url = self.RESULT_FALLBACK_URL.format(race_id=race_id)
                        kind = "result"
                    else:
                        url = self.CARD_URL.format(race_id=race_id)
                        kind = "card"
                    cache_path = self._cache_path(
                        dataset_type, storage_key, kind, race_id
                    )
                    html = self._download(
                        url,
                        cache_path,
                        force=force,
                    )
                    metadata = self._metadata(
                        BeautifulSoup(html, "lxml"),
                        html,
                        race_id,
                        target_date,
                    )
                    horses = self._extract_horse_links(
                        BeautifulSoup(html, "lxml")
                    )
                    for horse_id, horse_name in dict(
                        horses
                    ).items():
                        try:
                            self._horse_pedigree(
                                horse_id,
                                storage_key,
                                dataset_type=dataset_type,
                                horse_name=horse_name,
                                download=True,
                                force=force,
                            )
                            collected_horse_ids.add(horse_id)
                        except Exception as exc:  # noqa: BLE001
                            self.logger.error(
                                f"{horse_id}: "
                                f"血統HTML取得・解析失敗: {exc}"
                            )
                    self._rename_race_html(
                        cache_path,
                        race_id,
                        target_date,
                        metadata.get("race_name"),
                        metadata.get("course_name"),
                        metadata.get("race_number"),
                    )
                    race_count += 1
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(
                        f"{target_date} {race_id}: HTML取得失敗: {exc}"
                    )
                if progress_callback:
                    fraction = (
                        (date_index - 1)
                        + race_index / max(len(race_ids), 1)
                    ) / total_dates
                    progress_callback(min(float(fraction), 1.0))
            if progress_callback:
                progress_callback(date_index / total_dates)
        return {
            "date_count": list_count,
            "race_count": race_count,
            "horse_count": len(collected_horse_ids),
        }

    def parse_cached_date_range(
        self,
        start_date: date,
        end_date: date,
        dataset_type: str,
        progress_callback=None,
    ) -> pd.DataFrame:
        """保存済みHTMLだけを解析する。ネットワークアクセスは行わない。"""
        if end_date < start_date:
            raise ValueError("終了日は開始日以降にしてください。")
        dates = [
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        ]
        frames: list[pd.DataFrame] = []
        kind = "result" if dataset_type == "historical" else "card"
        for index, target_date in enumerate(dates, start=1):
            date_text = target_date.strftime("%Y%m%d")
            list_path = (
                self._html_root(dataset_type) / str(target_date.year)
                / "race_list_db" / f"{date_text}.html"
            )
            if not list_path.exists():
                if progress_callback:
                    progress_callback(index / len(dates))
                continue
            list_html = list_path.read_text(encoding="utf-8", errors="replace")
            for race_id in self._extract_race_ids(list_html, date_text):
                candidates = sorted(
                    (
                        self._html_root(dataset_type)
                        / str(target_date.year)
                        / kind
                    ).glob(
                        f"{race_id}*.html"
                    )
                )
                if not candidates:
                    continue
                try:
                    html = candidates[-1].read_text(
                        encoding="utf-8", errors="replace"
                    )
                    frame = self._parse_page(
                        html, race_id, target_date, dataset_type
                    )
                    frame = self._add_pedigrees(
                        frame,
                        str(target_date.year),
                        dataset_type=dataset_type,
                        download=False,
                    )
                    if not frame.empty:
                        frames.append(frame)
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(
                        f"{target_date} {race_id}: 保存HTML解析失敗: {exc}"
                    )
            if progress_callback:
                progress_callback(index / len(dates))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_card(self, race_id: str, race_date: date, run_id: str, force: bool = False) -> pd.DataFrame:
        url = self.CARD_URL.format(race_id=race_id)
        self.logger.info(f"出馬表解析対象URL: {url}")
        html = self._download(
            url,
            self._cache_path("upcoming", run_id, "card", race_id),
            force=force,
        )
        frame = self._parse_page(
            html,
            race_id,
            race_date,
            dataset_type="upcoming",
        )
        return self._add_pedigrees(
            frame,
            str(race_date.year),
            dataset_type="upcoming",
            download=True,
            force=force,
        )

    def fetch_result(
        self,
        race_id: str,
        race_date: date,
        run_id: str,
        force: bool = False,
    ) -> pd.DataFrame:
        """
        過去レース結果は旧DBページだけを取得する。

        保存先:
            raw_html/historical/<year>/result/<race_id>.html
        """
        url = self.RESULT_FALLBACK_URL.format(race_id=race_id)
        cache = self._cache_path(
            "historical", run_id, "result", race_id
        )

        self.logger.info(f"過去結果解析対象URL: {url}")

        html = self._download(
            url,
            cache,
            force=force,
        )

        frame = self._parse_page(
            html,
            race_id,
            race_date,
            dataset_type="historical",
        )

        race_name = (
            frame["race_name"].dropna().iloc[0]
            if not frame.empty
            and "race_name" in frame.columns
            and frame["race_name"].notna().any()
            else "レース名不明"
        )
        course_name = (
            frame["course_name"].dropna().iloc[0]
            if not frame.empty
            and "course_name" in frame.columns
            and frame["course_name"].notna().any()
            else "開催地不明"
        )
        race_number = (
            frame["race_number"].dropna().iloc[0]
            if not frame.empty
            and "race_number" in frame.columns
            and frame["race_number"].notna().any()
            else None
        )

        self._rename_race_html(
            cache,
            race_id,
            race_date,
            race_name,
            course_name,
            race_number,
        )
        return self._add_pedigrees(
            frame,
            str(race_date.year),
            dataset_type="historical",
            download=True,
            force=force,
        )

    def _select_table(self, html: str, dataset_type: str) -> pd.DataFrame:
        tables = pd.read_html(StringIO(html))
        scored: list[tuple[int, pd.DataFrame]] = []
        for table in tables:
            table = table.copy()
            table.columns = _flatten_columns(table.columns)
            columns_text = "|".join(map(str, table.columns))
            score = 0
            if "馬名" in columns_text:
                score += 4
            if "騎手" in columns_text:
                score += 2
            if "馬番" in columns_text:
                score += 2
            if dataset_type == "historical" and "着順" in columns_text:
                score += 5
            if dataset_type == "upcoming" and "着順" not in columns_text:
                score += 1
            if len(table) >= 5:
                score += 1
            scored.append((score, table))
        if not scored:
            return pd.DataFrame()
        best_score, best_table = max(scored, key=lambda item: item[0])
        if best_score < 5:
            return pd.DataFrame()
        return best_table

    def _metadata(
        self,
        soup: BeautifulSoup,
        html: str,
        race_id: str,
        race_date: date,
    ) -> dict[str, object]:
        """
        レース名・馬場種別・距離などは馬柱ではなく、
        ページ上部のレース情報領域から取得する。
        """
        intro = soup.select_one(
            ".data_intro, .RaceData01, .RaceData02, "
            ".race_head_inner, .RaceList_NameBox"
        )

        race_name_node = soup.select_one(
            ".data_intro h1, .RaceName, .race_name, "
            ".RaceList_NameBox h1, h1"
        )
        race_name = (
            race_name_node.get_text(" ", strip=True)
            if race_name_node
            else ""
        )

        intro_text = (
            intro.get_text(" ", strip=True)
            if intro
            else ""
        )
        title_text = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else ""
        )

        # 旧DBでは開催場所や日付がsmalltxt側にある場合がある。
        small_text_nodes = soup.select(
            ".data_intro .smalltxt, .smalltxt, "
            ".RaceData02, .RaceData01"
        )
        small_text = " ".join(
            node.get_text(" ", strip=True)
            for node in small_text_nodes
        )

        page_text = soup.get_text(" ", strip=True)
        combined = " ".join(
            part
            for part in [
                intro_text,
                small_text,
                page_text,
                title_text,
            ]
            if part
        )
        combined = re.sub(r"\s+", " ", combined)

        # ページ上部のレース条件を取得する。
        # 例:
        # 芝右1200m / 天候 : 晴 / 芝 : 良
        # ダ左1800m / 天候：曇 / ダート：稍重
        # 障芝2860m / 天候 : 晴 / 芝 : 良
        condition_patterns = [
            re.compile(
                r"(?P<surface>芝|ダート|ダ|障害|障)"
                r"(?:直|右|左|外|内|芝|ダート|ダ|障|\s)*"
                r"(?P<distance>\d{3,4})\s*[mｍ]",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"(?P<distance>\d{3,4})\s*[mｍ]"
                r"[^。/|]{0,20}"
                r"(?P<surface>芝|ダート|ダ|障害|障)",
                flags=re.IGNORECASE,
            ),
        ]

        surface_match = None
        for pattern in condition_patterns:
            surface_match = pattern.search(combined)
            if surface_match:
                break

        surface = None
        distance = None
        if surface_match:
            surface_raw = surface_match.group("surface")
            surface = {
                "ダ": "ダート",
                "障": "障害",
            }.get(surface_raw, surface_raw)
            distance = float(surface_match.group("distance"))

        weather_patterns = [
            r"天候\s*[:：]\s*([^/|\s]+)",
            r"天候\s+([^/|\s]+)",
        ]
        weather = None
        for pattern in weather_patterns:
            match = re.search(pattern, combined)
            if match:
                weather = match.group(1).strip()
                break

        track_patterns = [
            r"(?:芝|ダート|ダ|馬場)\s*[:：]\s*"
            r"(良|稍重|重|不良)",
            r"馬場状態\s*[:：]\s*(良|稍重|重|不良)",
        ]
        track_condition = None
        for pattern in track_patterns:
            match = re.search(pattern, combined)
            if match:
                track_condition = match.group(1).strip()
                break

        # HTMLタグ内に条件が分割され、テキスト結合で拾えない場合の
        # 最終フォールバック。
        if surface is None or distance is None:
            html_text = re.sub(r"<[^>]+>", " ", html)
            html_text = re.sub(r"&nbsp;|&#160;", " ", html_text)
            html_text = re.sub(r"\s+", " ", html_text)

            for pattern in condition_patterns:
                surface_match = pattern.search(html_text)
                if surface_match:
                    surface_raw = surface_match.group("surface")
                    surface = {
                        "ダ": "ダート",
                        "障": "障害",
                    }.get(surface_raw, surface_raw)
                    distance = float(
                        surface_match.group("distance")
                    )
                    break

        if weather is None:
            for pattern in weather_patterns:
                match = re.search(pattern, html)
                if match:
                    weather = match.group(1).strip()
                    break

        if track_condition is None:
            for pattern in track_patterns:
                match = re.search(pattern, html)
                if match:
                    track_condition = match.group(1).strip()
                    break
        # 開催地はrace_idの競馬場コード（5～6桁目）から確定する。
        # ページ全体にはナビゲーション等で他場名も含まれるため、
        # 文字列検索では中山・東京などを誤取得することがある。
        course_code = (
            race_id[4:6]
            if len(race_id) >= 6
            else ""
        )
        course_name = JRA_COURSE_CODE_MAP.get(course_code)

        # race_idで判定できない場合だけページ上部情報から補完する。
        if course_name is None:
            course_name = next(
                (
                    course
                    for course in COURSES
                    if course in intro_text
                    or course in small_text
                ),
                None,
            )

        race_number = (
            int(race_id[-2:])
            if race_id[-2:].isdigit()
            else None
        )

        # titleしか取れない場合はnetkeiba固有の後置文言を除去する。
        if not race_name and title_text:
            race_name = title_text

        # ページタイトル由来の後置情報を除去する。
        # 例:
        # 3歳未勝利｜2026年2月22日 _ 競馬データベース - netkeiba
        # -> 3歳未勝利
        if race_name:
            race_name = str(race_name).strip()

            # 全角・半角の縦棒以降をすべて削除する。
            race_name = re.sub(
                r"[｜|].*$",
                "",
                race_name,
            ).strip()

            # 縦棒がないタイトル形式にも対応する。
            race_name = re.sub(
                r"\s*[_＿]\s*競馬データベース.*$",
                "",
                race_name,
                flags=re.IGNORECASE,
            ).strip()
            race_name = re.sub(
                r"\s*-\s*netkeiba.*$",
                "",
                race_name,
                flags=re.IGNORECASE,
            ).strip()

        self.logger.info(
            "レース上部情報: "
            f"race_id={race_id}, "
            f"race_name={race_name or None}, "
            f"surface={surface}, "
            f"distance={distance}, "
            f"weather={weather}, "
            f"track_condition={track_condition}, "
            f"course_code={course_code}, "
            f"course_name={course_name}"
        )

        return {
            "race_id": race_id,
            "race_date": race_date,
            "course_name": course_name,
            "race_number": race_number,
            "race_name": race_name or None,
            "surface": surface,
            "distance": distance,
            "weather": weather,
            "track_condition": track_condition,
        }

    def _rename_race_html(
        self,
        cache_path: Path,
        race_id: str,
        race_date: date,
        race_name: object,
        course_name: object,
        race_number: object,
    ) -> Path:
        """
        レースHTMLを
        race_id_開催地_年月日_Rxx_レース名.html
        へ変更する。
        """
        date_text = race_date.strftime("%Y年%m月%d日")
        safe_race_name = _safe_filename_part(race_name)
        safe_course_name = _safe_filename_part(
            course_name or "開催地不明",
            max_length=20,
        )
        try:
            race_no = int(race_number)
            race_number_text = f"R{race_no:02d}"
        except (TypeError, ValueError):
            race_number_text = "R不明"

        target = cache_path.with_name(
            f"{race_id}_{safe_course_name}_{date_text}_"
            f"{race_number_text}_{safe_race_name}.html"
        )

        if cache_path.resolve() == target.resolve():
            return target

        # キャッシュ利用時は _download 内で既存の名称付きHTMLを読むため、
        # 呼び出し元のrace_idだけのパスが存在しない場合がある。
        if not cache_path.exists():
            candidates = sorted(
                cache_path.parent.glob(f"{race_id}*.html"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                return cache_path
            cache_path = candidates[0]
            if cache_path.resolve() == target.resolve():
                return target

        # 同名ファイルが既にあれば、今回の正規化済みHTMLで置き換える。
        if target.exists():
            target.unlink()

        cache_path.replace(target)
        self.logger.info(
            f"HTMLファイル名変更: {cache_path.name} -> {target.name}"
        )
        return target

    def _rename_horse_html(
        self,
        cache_path: Path,
        horse_id: str,
        horse_name: object,
    ) -> Path:
        safe_horse_name = _safe_filename_part(
            horse_name or "競走馬名不明"
        )
        target = cache_path.with_name(
            f"{horse_id}_{safe_horse_name}.html"
        )

        if not cache_path.exists():
            candidates = sorted(
                cache_path.parent.glob(f"{horse_id}*.html"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                return cache_path
            cache_path = candidates[0]

        if cache_path.resolve() == target.resolve():
            return target
        if target.exists():
            target.unlink()

        cache_path.replace(target)
        self.logger.info(
            f"競走馬HTMLファイル名変更: "
            f"{cache_path.name} -> {target.name}"
        )
        return target

    @staticmethod
    def _parse_pedigree_html(
        html: str,
    ) -> dict[str, str | None]:
        """Parse sire, dam and damsire from netkeiba's pedigree table."""
        empty = {
            "sire_id": None,
            "sire_name": None,
            "dam_id": None,
            "dam_name": None,
            "damsire_id": None,
            "damsire_name": None,
        }
        soup = BeautifulSoup(html, "lxml")
        table = soup.select_one(
            "table.blood_table, table.pedigree_table"
        )
        if table is None:
            return empty

        grid: dict[tuple[int, int], object] = {}
        rows = table.find_all("tr")
        for row_index, row in enumerate(rows):
            column_index = 0
            cells = row.find_all(["td", "th"], recursive=False)
            for cell in cells:
                while (row_index, column_index) in grid:
                    column_index += 1
                try:
                    rowspan = max(int(cell.get("rowspan", 1)), 1)
                except (TypeError, ValueError):
                    rowspan = 1
                try:
                    colspan = max(int(cell.get("colspan", 1)), 1)
                except (TypeError, ValueError):
                    colspan = 1
                for row_offset in range(rowspan):
                    for column_offset in range(colspan):
                        grid[
                            (
                                row_index + row_offset,
                                column_index + column_offset,
                            )
                        ] = cell
                column_index += colspan

        def pedigree_value(
            row_index: int,
            column_index: int,
        ) -> tuple[str | None, str | None]:
            cell = grid.get((row_index, column_index))
            if cell is None:
                return None, None
            anchor = cell.find(
                "a",
                href=re.compile(r"/horse/(?:ped/)?[0-9a-zA-Z]+"),
            )
            if anchor is None:
                return None, None
            href = str(anchor.get("href", ""))
            match = re.search(
                r"/horse/(?:ped/)?([0-9a-zA-Z]+)",
                href,
            )
            name = anchor.get_text(" ", strip=True) or None
            return (match.group(1) if match else None), name

        sire_cell = grid.get((0, 0))
        try:
            dam_row = max(
                int(sire_cell.get("rowspan", 1)),
                1,
            )
        except (AttributeError, TypeError, ValueError):
            dam_row = max(len(rows) // 2, 1)

        sire_id, sire_name = pedigree_value(0, 0)
        dam_id, dam_name = pedigree_value(dam_row, 0)
        damsire_id, damsire_name = pedigree_value(dam_row, 1)
        return {
            "sire_id": sire_id,
            "sire_name": sire_name,
            "dam_id": dam_id,
            "dam_name": dam_name,
            "damsire_id": damsire_id,
            "damsire_name": damsire_name,
        }

    def _cached_pedigree_path(
        self,
        horse_id: str,
        dataset_type: str,
        *,
        include_other: bool = True,
    ) -> Path | None:
        roots = [self._html_root(dataset_type)]
        if include_other:
            other_type = (
                "upcoming"
                if dataset_type == "historical"
                else "historical"
            )
            roots.append(self._html_root(other_type))
        for root in roots:
            candidates = sorted(
                (root / "horse").glob(f"{horse_id}*.html"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0]
        return None

    def _horse_pedigree(
        self,
        horse_id: str,
        storage_key: str,
        *,
        dataset_type: str,
        horse_name: str | None = None,
        download: bool,
        force: bool = False,
    ) -> dict[str, str | None]:
        cache_key = (dataset_type, horse_id)
        if cache_key in self.pedigree_cache:
            return self.pedigree_cache[cache_key]

        if download:
            url = self.HORSE_PEDIGREE_URL.format(
                horse_id=horse_id
            )
            cache_path = self._cache_path(
                dataset_type,
                storage_key,
                "horse",
                horse_id,
            )
            other_cache = (
                self._cached_pedigree_path(
                    horse_id,
                    dataset_type,
                    include_other=True,
                )
                if not force
                else None
            )
            if (
                other_cache is not None
                and other_cache.parent
                != cache_path.parent
            ):
                cache_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                shutil.copy2(other_cache, cache_path)
                html = cache_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                self.logger.info(
                    "別用途の保存済み競走馬HTMLを再利用: "
                    f"{other_cache} -> {cache_path}"
                )
            else:
                html = self._download(
                    url,
                    cache_path,
                    force=force,
                )
            self._rename_horse_html(
                cache_path,
                horse_id,
                horse_name,
            )
        else:
            path = self._cached_pedigree_path(
                horse_id,
                dataset_type,
            )
            if path is None:
                pedigree = self._parse_pedigree_html("")
                self.pedigree_cache[cache_key] = pedigree
                return pedigree
            html = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        pedigree = self._parse_pedigree_html(html)
        self.pedigree_cache[cache_key] = pedigree
        return pedigree

    def _add_pedigrees(
        self,
        frame: pd.DataFrame,
        storage_key: str,
        *,
        dataset_type: str,
        download: bool,
        force: bool = False,
    ) -> pd.DataFrame:
        result = frame.copy()
        pedigree_columns = [
            "sire_id",
            "sire_name",
            "dam_id",
            "dam_name",
            "damsire_id",
            "damsire_name",
        ]
        if result.empty:
            for column in pedigree_columns:
                result[column] = pd.Series(dtype="string")
            return result

        pedigrees: dict[str, dict[str, str | None]] = {}
        horse_names = (
            result.dropna(subset=["horse_id"])
            .assign(horse_id=lambda frame: frame["horse_id"].astype(str))
            .drop_duplicates("horse_id")
            .set_index("horse_id")["horse_name"]
            .to_dict()
            if {"horse_id", "horse_name"} <= set(result.columns)
            else {}
        )
        horse_ids = (
            result["horse_id"].dropna().astype(str).unique()
            if "horse_id" in result.columns
            else []
        )
        for horse_id in horse_ids:
            try:
                pedigrees[horse_id] = self._horse_pedigree(
                    horse_id,
                    storage_key,
                    dataset_type=dataset_type,
                    horse_name=(
                        str(horse_names[horse_id])
                        if horse_id in horse_names
                        and pd.notna(horse_names[horse_id])
                        else None
                    ),
                    download=download,
                    force=force,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.error(
                    f"{horse_id}: 血統HTML取得・解析失敗: {exc}"
                )

        for column in pedigree_columns:
            result[column] = result["horse_id"].map(
                lambda value: (
                    pedigrees.get(str(value), {}).get(column)
                    if pd.notna(value)
                    else None
                )
            )
        return result

    @staticmethod
    def _extract_horse_links(
        soup: BeautifulSoup,
    ) -> list[tuple[str, str]]:
        horses: list[tuple[str, str]] = []
        rows = soup.select(
            "tr.HorseList, table.RaceTable01 tr, "
            "table.race_table_01 tr"
        )
        for row in rows:
            if not row.find_all("td"):
                continue
            anchor = row.find(
                "a",
                href=re.compile(r"/horse/[0-9a-zA-Z]+"),
            )
            if anchor is None:
                continue
            match = re.search(
                r"/horse/([0-9a-zA-Z]+)",
                str(anchor.get("href", "")),
            )
            horse_name = anchor.get_text(" ", strip=True)
            if match and horse_name:
                horses.append((match.group(1), horse_name))
        return horses

    def _extract_link_ids(self, soup: BeautifulSoup) -> tuple[list[str | None], list[str | None], list[str | None]]:
        rows = soup.select("tr.HorseList, table.RaceTable01 tr, table.race_table_01 tr")
        horse_ids: list[str | None] = []
        jockey_ids: list[str | None] = []
        trainer_ids: list[str | None] = []
        for row in rows:
            if not row.find_all("td"):
                continue
            html = str(row)
            horse = re.search(r"/horse/([0-9a-zA-Z]+)", html)
            jockey = re.search(r"/jockey/(?:result/recent/)?(\d+)", html)
            trainer = re.search(r"/trainer/(?:result/recent/)?(\d+)", html)
            horse_ids.append(horse.group(1) if horse else None)
            jockey_ids.append(jockey.group(1) if jockey else None)
            trainer_ids.append(trainer.group(1) if trainer else None)
        return horse_ids, jockey_ids, trainer_ids

    @staticmethod
    def _historical_positional_columns(table: pd.DataFrame) -> dict[str, str | None]:
        """
        旧DB結果表の列位置を使ったフォールバック。

        旧DBの標準列:
        0 着順 / 1 枠番 / 2 馬番 / 3 馬名 / 4 性齢 / 5 斤量 /
        6 騎手 / 7 タイム / 8 着差 / 10 通過 / 11 上り /
        12 単勝 / 13 人気 / 14 馬体重 / 18 調教師 / 19 馬主
        """
        columns = list(map(str, table.columns))

        def at(index: int) -> str | None:
            return columns[index] if index < len(columns) else None

        return {
            "finish_position": at(0),
            "frame_number": at(1),
            "horse_number": at(2),
            "horse_name": at(3),
            "sex_age": at(4),
            "carried_weight": at(5),
            "jockey_name": at(6),
            "time_seconds": at(7),
            "odds": at(9),
            "popularity": at(10),
            "body_weight": at(11),
            "trainer_name": at(12),
        }

    def _parse_page(self, html: str, race_id: str, race_date: date, dataset_type: str) -> pd.DataFrame:
        soup = BeautifulSoup(html, "lxml")
        table = self._select_table(html, dataset_type)
        if table.empty:
            raise ValueError("出走馬テーブルを認識できません。raw_htmlを確認してください。")

        table = table.dropna(how="all").reset_index(drop=True)
        metadata = self._metadata(soup, html, race_id, race_date)
        horse_ids, jockey_ids, trainer_ids = self._extract_link_ids(soup)

        column_candidates = {
            "finish_position": ["着順"],
            "frame_number": ["枠", "枠番"],
            "horse_number": ["馬番"],
            "horse_name": ["馬名"],
            "sex_age": ["性齢", "性年齢"],
            "carried_weight": ["斤量"],
            "jockey_name": ["騎手"],
            "trainer_name": ["調教師", "厩舎"],
            "time_seconds": ["タイム"],
            "odds": ["単勝", "オッズ"],
            "popularity": ["人気"],
            "body_weight": ["馬体重"],
        }
        columns = {
            key: _first_column(table, candidates)
            for key, candidates in column_candidates.items()
        }

        # 旧DB結果表では、pd.read_html()が列名を正しく取得できず、
        # 着順・馬番・人気などが欠損する場合がある。
        # historicalでは固定列位置をフォールバックとして利用する。
        if dataset_type == "historical":
            positional = self._historical_positional_columns(table)
            for key, fallback_column in positional.items():
                if columns.get(key) is None and fallback_column is not None:
                    columns[key] = fallback_column

            self.logger.info(
                "旧DB列対応: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in columns.items()
                )
            )

        if not columns["horse_name"]:
            raise ValueError(
                f"馬名列が見つかりません。列={list(table.columns)}"
            )

        required_historical = [
            "finish_position",
            "horse_number",
            "horse_name",
            "popularity",
        ]
        if dataset_type == "historical":
            unresolved = [
                key for key in required_historical
                if not columns.get(key)
            ]
            if unresolved:
                self.logger.warning(
                    "旧DB結果表で未解決の列: "
                    + ", ".join(unresolved)
                    + f" / 実列={list(table.columns)}"
                )

        records: list[dict[str, object]] = []
        for index, row in table.iterrows():
            horse_name = str(row[columns["horse_name"]]).strip()
            if not horse_name or horse_name == "nan" or horse_name == "馬名":
                continue
            sex, age = _parse_sex_age(row[columns["sex_age"]]) if columns["sex_age"] else (None, None)
            body_weight, body_weight_change = (
                _parse_body_weight(row[columns["body_weight"]]) if columns["body_weight"] else (None, None)
            )
            record = dict(metadata)
            record.update(
                {
                    "horse_id": horse_ids[index] if index < len(horse_ids) else None,
                    "horse_name": horse_name,
                    "horse_number": _to_float(row[columns["horse_number"]]) if columns["horse_number"] else None,
                    "frame_number": _to_float(row[columns["frame_number"]]) if columns["frame_number"] else None,
                    "sex": sex,
                    "age": age,
                    "carried_weight": _to_float(row[columns["carried_weight"]]) if columns["carried_weight"] else None,
                    "jockey_id": jockey_ids[index] if index < len(jockey_ids) else None,
                    "jockey_name": str(row[columns["jockey_name"]]).strip() if columns["jockey_name"] else None,
                    "trainer_id": trainer_ids[index] if index < len(trainer_ids) else None,
                    "trainer_name": str(row[columns["trainer_name"]]).strip() if columns["trainer_name"] else None,
                    "odds": _to_float(row[columns["odds"]]) if columns["odds"] else None,
                    "popularity": _to_float(row[columns["popularity"]]) if columns["popularity"] else None,
                    "body_weight": body_weight,
                    "body_weight_change": body_weight_change,
                    "finish_position": _finish_position(row[columns["finish_position"]]) if columns["finish_position"] else None,
                    "time_seconds": _time_to_seconds(row[columns["time_seconds"]]) if columns["time_seconds"] else None,
                    "dataset_type": dataset_type,
                }
            )
            records.append(record)
        return pd.DataFrame(records)
