# JRA公式の結果画面を取得・保存し、着順、結果状態、100円当たり払戻金、返還対象を抽出する。
# 取消・除外・競走中止などは数値着順へ推測変換せず、状態として区別する。
# 一時的な通信障害は待機して再試行し、保存済みHTMLが使える場合は再利用する。
#
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

# プロジェクト既定の保存先。DBを差し替えるテストではこの参照も隔離先へ変更する。
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


# JRA公式結果の取得・再試行・キャッシュ再利用と、照合用の解析を受け持つ。
class JraResultFetcher:
    """JRA公式のレース結果HTMLを保存し、着順と払戻金を解析する。"""

    URL = "https://www.jra.go.jp/JRADB/accessS.html"
    ENTRY_CNAME = "pw01sli00/AF"
    RETRY_DELAYS = (10, 30, 60)
    RETRY_STATUS_CODES = {500, 502, 503, 504}
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

    # 結果HTMLの保存先と通信条件を設定し、画面遷移先を再利用するための内部情報を準備する。
    def __init__(
        self,
        logger,
        interval_seconds: float = 2.0,
        timeout: int = 30,
        session: requests.Session | None = None,
        output_root: Path | None = None,
    ) -> None:
        # 取得や解析の経過・警告・例外を通知するロガー。
        self.logger = logger
        # 取得先への連続アクセスを避けるための要求間隔（秒）。
        self.interval_seconds = max(float(interval_seconds), 0.0)
        # 一回のHTTP要求に対する待機上限秒数。
        self.timeout = timeout
        # 再利用するHTTPセッション。テストでは偽セッションへ差し替える。
        self.session = session or requests.Session()
        # 取得HTMLを保存する基準ディレクトリ。
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

    # JRAページのリンクやイベント属性から、POST遷移に使う画面識別子を抽出する。
    @classmethod
    def _action_cnames(cls, html: str) -> list[str]:
        return list(dict.fromkeys(cls._ACTION_PATTERN.findall(html)))

    # 一時障害のときだけ待機を延ばして再試行する。成功応答を得るまで正常HTMLとして保存しない。
    def _post(self, cname: str) -> str:
        # These POSTs only retrieve result pages; retry the same page on transient failures.
        for attempt in range(len(self.RETRY_DELAYS) + 1):
            try:
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
                break
            except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as exc:
                if isinstance(exc, requests.HTTPError):
                    status = exc.response.status_code if exc.response is not None else None
                    if exc.response is not None:
                        exc.response.close()
                    retryable = status in self.RETRY_STATUS_CODES
                else:
                    retryable = not isinstance(exc, requests.exceptions.SSLError)
                # 恒久エラー、または再試行上限なら待機を続けず元の例外を上位へ返す。
                if not retryable or attempt == len(self.RETRY_DELAYS):
                    raise
                delay = self.RETRY_DELAYS[attempt]
                self.logger.warning(
                    "JRA結果取得の一時エラー: cname=%s, %s秒後に再試行 (%s/%s): %s",
                    cname, delay, attempt + 1, len(self.RETRY_DELAYS), exc,
                )
                time.sleep(delay)
        response.encoding = "cp932"
        html = response.text
        # HTTP成功でもJRAのエラー画面なら結果として扱わず、保存前に失敗させる。
        if "パラメータエラー" in html or "ＤＢ検索エラー" in html:
            raise ValueError("JRA結果ページへのアクセスでエラーが返されました。")
        if self.interval_seconds:
            time.sleep(self.interval_seconds)
        return html

    # HTML内の文字コード宣言をUTF-8へそろえて保存し、再解析時のデコードを安定させる。
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

    # 12桁のレースIDを開催年・場・回・日・レース番号へ分解し、対象日との整合性も確認する。
    @staticmethod
    def _race_parts(race_id: str, race_date: date) -> dict[str, str]:
        # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
        # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
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
        # レースIDの開催年と指定日付の年が異なる入力は、別レースへの誤遷移を避けるため拒否する。
        if parts["year"] != str(race_date.year):
            raise ValueError("race_idの年と比較対象日が一致しません。")
        return parts

    # 開催一覧から開催場・日・レースに対応する公式の画面遷移先を解決する。
    def _discover_race_cname(self, race_id: str, race_date: date) -> str:
        # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
        # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
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
        # 対象競馬場のリンクが見つからなければ、別の場の結果を取得しない。
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
        # 目的のレースの公式遷移先が存在することを確認し、欠落を結果未取得として報告する。
        if race_id not in self._race_cnames:
            raise ValueError(f"JRAの開催ページに対象レースがありません: {race_id}")
        return self._race_cnames[race_id]

    # 結果HTMLのキャッシュを使うか取得し直すかを選び、着順と公式精算情報を持つ表へ解析する。
    def fetch_result_for_comparison(
        self,
        race_id: str,
        race_date: date,
        *,
        force: bool = True,
    ) -> pd.DataFrame:
        """JRA公式結果を保存して解析する。force=Falseなら保存HTMLを再利用する。"""
        # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
        # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
        # force: キャッシュを使わず新規取得するかを指定する。
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
        # 結果表が空、または数値着順が一頭もない状態を確定結果として扱わない。
        if frame.empty or frame["finish_position"].notna().sum() == 0:
            raise ValueError(
                "JRAから確定着順を取得できませんでした。結果確定後に再実行してください。"
            )
        return frame


# 表示文字列から整数を読み取り、解析不能な値は欠損として返す。
def _integer(text: str) -> int | None:
    normalized = unicodedata.normalize("NFKC", text).replace(",", "")
    match = re.search(r"\d+", normalized)
    return int(match.group()) if match else None


# 買い目の表示から選択番号を取り出し、券種の順序条件に合わせた形式へそろえる。
def _selection(text: str, *, unordered: bool) -> str:
    values = [int(value) for value in re.findall(
        r"\d+", unicodedata.normalize("NFKC", text)
    )]
    if unordered:
        values.sort()
    return "-".join(map(str, values))


# 着順欄を数値順位と結果状態へ分離し、取消・除外・中止を順位へ推測変換しない。
def _finish(text: str) -> tuple[int | None, str]:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if re.fullmatch(r"[1-9]\d*", normalized):
        return int(normalized), "finished"
    return None, {
        "取消": "scratched", "除外": "excluded", "中止": "did_not_finish",
        "失格": "disqualified",
    }.get(normalized, "unknown")


# 馬番返還・枠番返還・同枠組合せだけの返還を別々に記録し、券種の精算に利用する。
def _official_refunds(soup: BeautifulSoup) -> dict[str, list[int]]:
    """Read horse refunds, whole-frame refunds and same-frame-only refunds."""
    refunds = {"horse_numbers": [], "frame_numbers": [], "same_frame_numbers": []}
    fields = {"馬番": "horse_numbers", "枠番": "frame_numbers",
              "同枠": "same_frame_numbers"}
    for node in soup.select(".refund_area .restoration dd"):
        text = unicodedata.normalize("NFKC", node.get_text(" ", strip=True))
        sections = re.split(r"返還(馬番|枠番|同枠)", text)
        if sections[0].strip() or len(sections) == 1:
            raise ValueError(f"公式返還情報を解析できません: {text}")
        for label, values in zip(sections[1::2], sections[2::2]):
            unit = "番" if label == "馬番" else "枠"
            if not re.fullmatch(rf"[\s\d{unit}、,・]+", values) or not re.search(r"\d", values):
                raise ValueError(f"公式返還情報を解析できません: {text}")
            numbers = [int(value) for value in re.findall(r"\d+", values)]
            if any(n < 1 or n > (18 if label == "馬番" else 8) for n in numbers):
                raise ValueError(f"公式返還番号が不正です: {text}")
            refunds[fields[label]].extend(numbers)
    return {key: sorted(set(values)) for key, values in refunds.items()}


# 確定結果表と払戻欄を解析する。返還情報と払戻金はDataFrameのattrsへ付けて照合側へ渡す。
def parse_jra_result_html(path: Path, race_id: str) -> pd.DataFrame:
    """保存済みJRA結果HTMLから着順と100円当たりの公式払戻金を読む。"""
    # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
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
            place = tr.select_one("td.place")
            finish_position, result_status = _finish(place.get_text(" ", strip=True) if place else "")
            rows.append({
                "race_id": race_id,
                "horse_id": pd.NA,
                "horse_number": horse_number,
                "horse_name": tr.select_one("td.horse").get_text(" ", strip=True)
                if tr.select_one("td.horse") else pd.NA,
                "frame_number": frame_number,
                "finish_position": finish_position,
                "result_status": result_status,
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
        "frame_number", "finish_position", "result_status",
    ])
    frame.attrs["official_payouts"] = payouts
    frame.attrs["official_refunds"] = _official_refunds(soup)
    frame.attrs["source_html"] = str(path)
    return frame
