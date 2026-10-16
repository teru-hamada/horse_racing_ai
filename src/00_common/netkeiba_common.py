# netkeibaの開催一覧・出馬表・結果・血統を扱う共通実装。
# HTML取得と保存済みHTMLの解析を分離し、キャッシュを利用して再取得を減らす。
# サイト固有の表・リンク・文字列表現を、保存層で扱えるレース単位／出走馬単位の値へ変換する。
# 取得先の構造変更に対応する際は、正規の列名と位置による補完の両方を確認する。
#
from __future__ import annotations

import json
import re
import shutil
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from .config import PATHS
from .logging_utils import AppLogger


# 中央競馬の競馬場名。ダミーデータなどで使う名称の一覧。
COURSES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

# レースIDに含まれる場コードと、表示する競馬場名の対応。
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
# netkeibaのページ要求間隔（秒）。連続アクセスを抑える共通値。
PAGE_REQUEST_INTERVAL_SECONDS = 2.0


# 多段列名を文字列へまとめ、空白やUnnamedを整理して表の列候補を照合しやすくする。
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
    for index, col in enumerate(columns):
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
        if not normalized:
            normalized = f"column_{index}"
        flattened.append(normalized)

    return flattened


# 列名候補を優先順に照合する。サイト側の表記の揺れを吸収するため部分一致を使う。
def _first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        for col in df.columns:
            if candidate in str(col):
                return str(col)
    return None


# 桁区切りを除き、文字列中の数値部分を取り出す。数値が読めない場合はNoneとする。
def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


# 先頭の数字だけを順位として読む。取消など数字で始まらない表示から順位を推測しない。
def _finish_position(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    match = re.match(r"(\d+)", text)
    return float(match.group(1)) if match else None


# 通過順から最大4地点の整数を取り出し、不足地点はNoneで埋める。
def _passing_positions(value: object) -> tuple[int | None, ...]:
    """通過順を左から最大4地点に分割する。"""
    if value is None or pd.isna(value):
        return (None, None, None, None)

    # 着順を数値化したSeries。取消などは数値順位へ推測しない。
    positions = [
        int(position)
        for position in re.findall(r"\d+", str(value))[:4]
    ]
    return tuple(positions + [None] * (4 - len(positions)))


# 分と秒の表示を秒単位へ変換し、保存先のtime_secondsと単位をそろえる。
def _time_to_seconds(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not re.match(r"^\d+:\d+(?:\.\d+)?$", text):
        return None
    minute, second = text.split(":", 1)
    return float(minute) * 60 + float(second)


# 性齢の表示を性別と年齢へ分け、解析不能な値は欠損として扱う。
def _parse_sex_age(value: object) -> tuple[str | None, int | None]:
    text = "" if value is None else str(value).strip()
    match = re.search(r"([牡牝セ騙])\s*(\d+)", text)
    if not match:
        return None, None
    sex = "セ" if match.group(1) == "騙" else match.group(1)
    return sex, int(match.group(2))


# 馬体重と括弧内の増減を分ける。オッズなど他の数値抽出と混同しない。
def _parse_body_weight(value: object) -> tuple[float | None, float | None]:
    text = "" if value is None else str(value).strip()
    match = re.search(r"(\d+)\s*\(([+-]?\d+)\)", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    plain = re.search(r"(\d+)", text)
    return (float(plain.group(1)), None) if plain else (None, None)


# Windowsで使えないファイル名文字を除き、人手で見られる名前を安全に作る。
def _safe_filename_part(value: object, max_length: int = 80) -> str:
    """Windowsで使用できない文字を除去してファイル名用文字列を作る。"""
    text = "" if value is None else str(value).strip()
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text).strip(" ._")
    return text[:max_length] or "レース名不明"


# HTTPセッション、HTMLキャッシュ、サイト固有の解析をまとめた共通基底クラス。
class NetkeibaCommon:
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
    JRA_ODDS_API_URL = (
        "https://race.netkeiba.com/api/api_get_jra_odds.html"
    )

    # タイムアウト・取得間隔・ロガーとHTTPセッションを設定し、診断用の内部状態を初期化する。
    def __init__(self, logger: AppLogger, interval_seconds: float = 2.0, timeout: int = 30) -> None:
        # 取得や解析の経過・警告・例外を通知するロガー。
        self.logger = logger
        # 取得先への連続アクセスを避けるための要求間隔（秒）。
        self.interval_seconds = max(interval_seconds, 0.5)
        # 一回のHTTP要求に対する待機上限秒数。
        self.timeout = timeout
        # 再利用するHTTPセッション。テストでは偽セッションへ差し替える。
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

    # 過去結果と未開催出馬表で、参照するHTMLの基準保存先を切り替える。
    @staticmethod
    def _html_root(dataset_type: str) -> Path:
        # 過去結果と出馬表で、取得URLや列配置・必須情報が異なるため処理を切り替える。
        if dataset_type == "historical":
            return PATHS.historical_html
        # 未開催レースでは、静的な出馬表の補完や明示された取消状態を処理する。
        if dataset_type == "upcoming":
            return PATHS.upcoming_html
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    # データ種別・年・HTMLの種類・識別子で保存先を分け、異なるページの上書きを防ぐ。
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

    # HTMLのmeta宣言を保存実体と同じUTF-8へそろえ、古い宣言による再解釈を避ける。
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

    # 保存バイト列だけでなくHTML内の宣言もUTF-8へそろえ、後でブラウザで開いたときの文字化けを防ぐ。
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

    # 一部の馬だけのオッズで取得済みと判断せず、出走馬ごとの単勝と人気の補完状況を確認する。
    @staticmethod
    def _has_embedded_win_odds(html: str) -> bool:
        """全出走馬の単勝オッズ・人気がHTMLへ入っているか確認する。"""
        soup = BeautifulSoup(html, "lxml")
        odds_nodes = soup.select('[id^="odds-1_"]')
        if not odds_nodes:
            return False
        for odds_node in odds_nodes:
            suffix = str(odds_node.get("id", "")).removeprefix(
                "odds-1_"
            )
            popularity_node = soup.select_one(f"#ninki-1_{suffix}")
            odds_text = odds_node.get_text(strip=True)
            popularity_text = (
                popularity_node.get_text(strip=True)
                if popularity_node
                else ""
            )
            if (
                re.fullmatch(r"\d+(?:\.\d+)?", odds_text) is None
                or not popularity_text.isdigit()
            ):
                return False
        return True

    # 出馬表の対象要素へAPI値を埋め込む。取消などの明示状態はオッズで上書きしない。
    @staticmethod
    def _embed_win_odds(
        html: str,
        odds_rows: dict[str, object],
        official_datetime: str = "",
    ) -> tuple[str, int]:
        """公式APIの単勝オッズと人気を出馬表HTMLへ埋め込む。"""
        soup = BeautifulSoup(html, "lxml")
        embedded_count = 0
        for horse_number, values in odds_rows.items():
            if not isinstance(values, (list, tuple)) or len(values) < 3:
                continue
            suffix = str(horse_number).zfill(2)
            odds_node = soup.select_one(f"#odds-1_{suffix}")
            popularity_node = soup.select_one(f"#ninki-1_{suffix}")
            if odds_node is None or popularity_node is None:
                continue
            if any(node.get_text(strip=True) in {"取消", "出走取消", "除外", "競走除外"}
                   for node in (odds_node, popularity_node)):
                # Do not overwrite an explicit withdrawal with stale API odds.
                continue
            odds_text = str(values[0]).strip()
            popularity_text = str(values[2]).strip()
            if (
                re.fullmatch(r"\d+(?:\.\d+)?", odds_text) is None
                or not popularity_text.isdigit()
            ):
                continue
            odds_node.string = odds_text
            popularity_node.string = popularity_text
            embedded_count += 1

        if embedded_count:
            # 結合成功の行数を数える一時列。特徴量自体がNaNでもキーが一致したかを区別できる。
            marker = soup.new_string(
                "\nnetkeiba-odds-embedded: "
                f"{official_datetime or datetime.now().isoformat()}\n",
                Comment,
            )
            if soup.body is not None:
                soup.body.insert(0, marker)
        return str(soup), embedded_count

    # 静的HTMLだけでは不足するオッズを補い、後のDB作成を保存済みHTMLだけで完結できるようにする。
    def _enrich_upcoming_html_with_odds(
        self,
        html: str,
        race_id: str,
        cache_path: Path,
        force: bool = False,
    ) -> str:
        """
        出馬表の静的HTMLにない単勝オッズ・人気を公式APIから補完する。

        既に全頭分が埋め込まれている場合は、再取得指定時を除いて
        APIへアクセスしない。
        """
        # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
        # force: キャッシュを使わず新規取得するかを指定する。
        if self._has_embedded_win_odds(html) and not force:
            self.logger.info(
                f"{race_id}: 保存HTMLのオッズ・人気を再利用"
            )
            return html

        api_url = self.JRA_ODDS_API_URL
        self.current_url = api_url
        self.last_url = api_url
        if api_url not in self.requested_urls:
            self.requested_urls.append(api_url)
        try:
            response = self.session.get(
                api_url,
                params={
                    "pid": "api_get_jra_odds",
                    "input": "UTF-8",
                    "output": "json",
                    "race_id": race_id,
                    "type": "1",
                    "action": "init",
                    "sort": "odds",
                    "compress": "0",
                },
                headers={
                    "Referer": self.CARD_URL.format(race_id=race_id),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data")
            if isinstance(data, str) and data:
                data = json.loads(data)
            if not isinstance(data, dict):
                self.logger.info(
                    f"{race_id}: オッズ・人気はまだ公開されていません。"
                )
                return html
            odds = data.get("odds", {})
            win_odds = (
                odds.get("1", {})
                if isinstance(odds, dict)
                else {}
            )
            if not isinstance(win_odds, dict) or not win_odds:
                self.logger.info(
                    f"{race_id}: 単勝オッズ・人気はまだ公開されていません。"
                )
                return html
            enriched_html, embedded_count = self._embed_win_odds(
                html,
                win_odds,
                str(data.get("official_datetime", "")),
            )
            if not embedded_count:
                self.logger.warning(
                    f"{race_id}: オッズAPIに値がありますが、"
                    "HTMLの出走馬へ対応付けできませんでした。"
                )
                return html
            enriched_html = self._write_utf8_html(
                cache_path,
                enriched_html,
            )
            self.logger.info(
                f"{race_id}: 単勝オッズ・人気を"
                f"{embedded_count}頭分HTMLへ埋め込みました。"
            )
            return enriched_html
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                f"{race_id}: オッズAPI取得失敗。"
                f"出馬表HTMLのみ保存します: {exc}"
            )
            return html
        finally:
            time.sleep(self.interval_seconds)

    # 他の年別保存先にある同一ページを探し、血統など重複取得できるデータの通信を減らす。
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

    # 名称付与によってファイル名が変わった場合も、ID接頭辞から既存キャッシュを解決する。
    @staticmethod
    def _resolve_race_cache_path(cache_path: Path) -> Path:
        """IDだけの仮パスから、名称付与後の既存HTMLを解決する。"""
        if cache_path.exists():
            return cache_path
        candidates = sorted(
            cache_path.parent.glob(f"{cache_path.stem}*.html"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else cache_path

    # 現在のキャッシュ→既存保存分の再利用→HTTP取得の順に試す。forceではキャッシュを使わず取得する。
    # 取得先・文字コード・キャッシュ利用を診断へ残し、通信後はアクセス間隔を空ける。
    def _download(self, url: str, cache_path: Path, force: bool = False) -> str:
        # force: キャッシュを使わず新規取得するかを指定する。
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

        # 強制取得が要求されていない場合だけ、別の保存先にあるキャッシュも探す。
        if not force:
            existing_cache = self._find_existing_html(cache_path)
            # 再利用できるHTMLを発見したら今回の出力先へコピーし、新規要求は送らない。
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

    # ページ内の12桁の数字を無条件に採用せず、対象日の中央競馬のレース識別子へ絞る。
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

    # 対象日の表示グループ識別子をHTMLから取り出し、開催一覧の遷移に使う。
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

    # 開催日の一覧からレースIDを見つける。取得できない場合は診断を残して空一覧を返す。
    def race_ids_for_date(
        self,
        target_date: date,
        run_id: str,
        dataset_type: str,
        force: bool = False,
    ) -> list[str]:
        """対象日の中央競馬レースIDを取得する。"""
        # force: キャッシュを使わず新規取得するかを指定する。
        date_text = target_date.strftime("%Y%m%d")

        # 未開催レースでは、静的な出馬表の補完や明示された取消状態を処理する。
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
            # 対象日の開催グループが解決できない場合、別日の一覧へ進まず失敗させる。
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

    # 日付は両端を含める。レース単位の失敗を記録しながら、結果または出馬表・血統・オッズを収集する。
    def _collect_html_date_range(
        self,
        start_date: date,
        end_date: date,
        dataset_type: str,
        force: bool = False,
        progress_callback=None,
    ) -> dict[str, int]:
        """HTMLだけを取得する。解析やDB保存は行わない。"""
        # force: キャッシュを使わず新規取得するかを指定する。
        # 期間が逆転した入力は、空処理として見逃さず開始前に拒否する。
        if end_date < start_date:
            raise ValueError("終了日は開始日以降にしてください。")
        # 保存先と解析形式が決まらない種別は受け入れず、既知の二種類へ限定する。
        if dataset_type not in {"historical", "upcoming"}:
            raise ValueError(f"未対応のdataset_typeです: {dataset_type}")

        dates = [
            start_date + timedelta(days=i)
            for i in range((end_date - start_date).days + 1)
        ]
        list_count = 0
        race_count = 0
        odds_html_count = 0
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
                    # 過去結果と出馬表で、取得URLや列配置・必須情報が異なるため処理を切り替える。
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
                    # 未開催レースでは、静的な出馬表の補完や明示された取消状態を処理する。
                    if dataset_type == "upcoming":
                        cache_path = self._resolve_race_cache_path(
                            cache_path
                        )
                        html = self._enrich_upcoming_html_with_odds(
                            html,
                            race_id,
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
                # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
                if progress_callback:
                    race_phase_weight = (
                        0.75 if dataset_type == "upcoming" else 1.0
                    )
                    fraction = (
                        (date_index - 1)
                        + race_phase_weight
                        * race_index
                        / max(len(race_ids), 1)
                    ) / total_dates
                    progress_callback(min(float(fraction), 1.0))
            if dataset_type == "upcoming" and race_ids:
                # オッズは補助データである。未発売、JRA側の画面変更、通信失敗
                # などがあっても、従来の出馬表HTML収集は成功扱いにする。
                try:
                    from importlib import import_module

                    odds_module = import_module(
                        "src.20_scrapers_html_collection.jra_odds_html"
                    )
                    odds_collector = odds_module.JraOddsHtmlCollector(
                        logger=self.logger,
                        interval_seconds=self.interval_seconds,
                        timeout=self.timeout,
                    )
                    odds_html_count += odds_collector.collect_date(
                        target_date=target_date,
                        race_ids=race_ids,
                        output_directory=(
                            self._html_root(dataset_type)
                            / storage_key
                            / "odds"
                        ),
                        force=force,
                        progress_callback=(
                            (
                                lambda odds_progress: progress_callback(
                                    min(
                                        float(
                                            (date_index - 1)
                                            + 0.75
                                            + 0.25 * odds_progress
                                        )
                                        / total_dates,
                                        1.0,
                                    )
                                )
                            )
                            if progress_callback
                            else None
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(
                        f"{target_date}: JRAオッズHTMLを取得できませんでした。"
                        f"従来のレース情報のみ保持します: {exc}"
                    )
            # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
            if progress_callback:
                progress_callback(date_index / total_dates)
        return {
            "date_count": list_count,
            "race_count": race_count,
            "horse_count": len(collected_horse_ids),
            "odds_html_count": odds_html_count,
        }

    # 保存済み開催一覧から対象レースHTMLを探す。血統の追加もdownload=Falseで行い、通信を起こさない。
    def _parse_cached_date_range(
        self,
        start_date: date,
        end_date: date,
        dataset_type: str,
        progress_callback=None,
    ) -> pd.DataFrame:
        """保存済みHTMLだけを解析する。ネットワークアクセスは行わない。"""
        # 期間が逆転した入力は、空処理として見逃さず開始前に拒否する。
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
                # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
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
            # 呼出側が指定した場合だけ進捗を通知する。ジョブ実行では取消確認の契機にもなる。
            if progress_callback:
                progress_callback(index / len(dates))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # 主結果表を優先して一表ずつ解析し、壊れた払戻表などの補助表で全体が失敗するのを防ぐ。
    def _select_table(self, html: str, dataset_type: str) -> pd.DataFrame:
        # ページ全体を pd.read_html() に渡すと、古い保存HTMLに含まれる
        # 空または崩れた払戻テーブルなどで IndexError になり、正常な
        # レース結果表まで解析できないことがある。レース表を優先して
        # 1件ずつ解析し、解析不能な補助表は読み飛ばす。
        soup = BeautifulSoup(html, "lxml")
        table_nodes = soup.select("table.RaceTable01, table.race_table_01")
        if not table_nodes:
            table_nodes = soup.find_all("table")

        tables: list[pd.DataFrame] = []
        for table_node in table_nodes:
            try:
                # 旧DB結果表では通過・上り等が独自要素内にある。
                # 要素を展開し、表示されない有料指数列を除去して
                # ヘッダーとデータの列位置を一致させる。
                for wrapper in table_node.find_all("diary_snap_cut"):
                    wrapper.unwrap()
                for hidden_index in table_node.select(
                    '[class*="TimeIndex"]'
                ):
                    hidden_index.decompose()
                tables.extend(pd.read_html(StringIO(str(table_node))))
            except (IndexError, TypeError, ValueError) as exc:
                self.logger.warning(
                    "解析不能なHTMLテーブルをスキップしました: %s",
                    exc,
                )

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

    # レース全体に共通する開催条件を抽出し、各出走馬の行へ同じ情報を付与できるようにする。
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
        # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
        # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
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

    # レースIDを残したまま名称などをファイル名へ付け、人手でキャッシュを確認しやすくする。
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
        # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
        # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
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

    # 馬IDと表示名を用いて血統HTMLを識別しやすいファイル名へ変更する。
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

    # 血統表の構造から父・母・母父を読み分け、表示名とリンク先IDの両方を返す。
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

        # 血統表の指定位置から、馬名とリンク先IDを対応付けて取り出す。
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

    # 名称が付いた保存済み血統HTMLを馬IDから探し、既存キャッシュを再利用する。
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

    # 保存済み血統を優先し、downloadが許可された場合だけ不足ページを取得する。
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
        # force: キャッシュを使わず新規取得するかを指定する。
        cache_key = (dataset_type, horse_id)
        if cache_key in self.pedigree_cache:
            return self.pedigree_cache[cache_key]

        # 通信が許可された呼出しでだけ不足する血統HTMLを取得し、ローカル再解析の経路と区別する。
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

    # 出走馬に血統のIDと表示名を付ける。保存HTMLの再解析では追加通信をしない指定で利用する。
    def _add_pedigrees(
        self,
        frame: pd.DataFrame,
        storage_key: str,
        *,
        dataset_type: str,
        download: bool,
        force: bool = False,
    ) -> pd.DataFrame:
        # force: キャッシュを使わず新規取得するかを指定する。
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

    # 馬ページへのリンクからIDと名前を取り出し、血統取得の対象を列挙する。
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

    # 表の表示名とは別に、リンク先から馬・騎手・調教師の識別子を抽出する。
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

    # 旧結果表で列名が読めない場合だけ、既知の列位置から対応を補う。
    @staticmethod
    def _historical_positional_columns(table: pd.DataFrame) -> dict[str, str | None]:
        """
        旧DB結果表の列位置を使ったフォールバック。

        旧DBの標準列:
        0 着順 / 1 枠番 / 2 馬番 / 3 馬名 / 4 性齢 / 5 斤量 /
        6 騎手 / 7 タイム / 8 着差 / 9 通過 / 10 上り /
        11 単勝 / 12 人気 / 13 馬体重 / 17 調教師 / 18 馬主
        """
        columns = list(map(str, table.columns))

        # 既知の列位置が表の範囲内にある場合だけ列名を返し、列数の違いで添字例外を起こさない。
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
            "passing_positions": at(9),
            "last_3f_time": at(10),
            "odds": at(11),
            "popularity": at(12),
            "body_weight": at(13),
            "trainer_name": at(17),
        }

    # 現行出馬表の列名が不足した場合に、表の列位置を使って必要列を補う。
    @staticmethod
    def _upcoming_positional_columns(
        table: pd.DataFrame,
    ) -> dict[str, str | None]:
        """
        現行出馬表の列位置によるフォールバック。

        オッズ列は見出しが画像等で空になるため、列名だけでは
        特定できない場合がある。
        """
        columns = list(map(str, table.columns))

        # 既知の列位置が表の範囲内にある場合だけ列名を返し、列数の違いで添字例外を起こさない。
        def at(index: int) -> str | None:
            return columns[index] if index < len(columns) else None

        return {
            "frame_number": at(0),
            "horse_number": at(1),
            "horse_name": at(3),
            "sex_age": at(4),
            "carried_weight": at(5),
            "jockey_name": at(6),
            "trainer_name": at(7),
            "body_weight": at(8),
            "odds": at(9),
            "popularity": at(10),
        }

    # 主表・レース条件・リンクIDを統合し、一頭一行の保存用レコードへ変換する。
    # 出馬表の取消は明示ラベルから判定し、オッズ欠損だけでは取消扱いしない。
    def _parse_page(self, html: str, race_id: str, race_date: date, dataset_type: str) -> pd.DataFrame:
        # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
        # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
        soup = BeautifulSoup(html, "lxml")
        table = self._select_table(html, dataset_type)
        # 対象表を認識できない状態を正常なゼロ頭と混同しないように扱う。
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
            "passing_positions": ["通過"],
            "last_3f_time": ["上り", "上がり"],
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
        else:
            positional = self._upcoming_positional_columns(table)
            for key, fallback_column in positional.items():
                if columns.get(key) is None and fallback_column is not None:
                    columns[key] = fallback_column

        # 馬名列がなければ行を出走馬として解釈できないため、解析を止める。
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
        # 過去結果と出馬表で、取得URLや列配置・必須情報が異なるため処理を切り替える。
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
            passing_positions = (
                _passing_positions(row[columns["passing_positions"]])
                if columns["passing_positions"]
                else (None, None, None, None)
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
                    "last_3f_time": _to_float(row[columns["last_3f_time"]]) if columns["last_3f_time"] else None,
                    "passing_position_1": passing_positions[0],
                    "passing_position_2": passing_positions[1],
                    "passing_position_3": passing_positions[2],
                    "passing_position_4": passing_positions[3],
                    "dataset_type": dataset_type,
                }
            )
            # 未開催レースでは、静的な出馬表の補完や明示された取消状態を処理する。
            if dataset_type == "upcoming":
                # Only explicit withdrawal labels count; missing odds alone do not.
                labels = {str(value).strip() for value in row.values}
                record["entry_status"] = (
                    "scratched" if labels & {"取消", "出走取消"} else
                    "excluded" if labels & {"除外", "競走除外"} else "active"
                )
            records.append(record)
        return pd.DataFrame(records)
