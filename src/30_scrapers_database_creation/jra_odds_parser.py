# 保存済みJRAオッズHTMLを、券種・選択馬番／枠番・オッズ上下限の行へ正規化する。
# 単勝・複勝と組合せ式の表構造の違いを吸収し、後段のDB保存と買い目計算へ渡す。
# このモジュールはローカルHTMLを解析し、通信による取得は行わない。
#
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


# 券種別オッズの保存列。選択番号とオッズ上下限、収集情報を含む。
ODDS_COLUMNS = [
    "race_id", "race_date", "bet_type", "selection_1", "selection_2",
    "selection_3", "odds_min", "odds_max", "source_html",
]


# 桁区切りを除き、オッズ表示の小数値を読み取る。取消など数値でない表示はNoneにする。
def _number(value: object) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    return float(match.group()) if match else None


# 表示文字列から一つの選択番号を整数で取り出す。番号がなければNoneを返す。
def _selection(value: object) -> int | None:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else None


# 一つのオッズ情報を、保存列に対応するレコードへ変換する。
def _row(
    race_id: str,
    race_date: date,
    bet_type: str,
    selections: list[int | None],
    odds_min: float | None,
    odds_max: float | None,
    path: Path,
) -> dict[str, object] | None:
    # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    if odds_min is None or selections[0] is None:
        return None
    # 券種の選択番号を最大3枠へそろえ、未使用の番号列はNoneで埋める。
    padded = selections[:3] + [None] * (3 - len(selections))
    return {
        "race_id": race_id,
        "race_date": race_date,
        "bet_type": bet_type,
        "selection_1": padded[0],
        "selection_2": padded[1],
        "selection_3": padded[2],
        "odds_min": odds_min,
        "odds_max": odds_max if odds_max is not None else odds_min,
        "source_html": str(path),
    }


# 券種に応じて選択番号と上下限オッズを抽出し、取消や数値でない表示を無理にオッズへ変換しない。
def parse_odds_html(path: Path, race_date: date) -> pd.DataFrame:
    """保存済みのJRA券種別HTMLを正規化されたオッズ行へ変換する。"""
    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    match = re.fullmatch(
        r"(?P<race_id>\d{12})_(?P<bet_type>[a-z_]+)\.html", path.name
    )
    if match is None:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    race_id = match.group("race_id")
    bet_type = match.group("bet_type")
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    rows: list[dict[str, object]] = []

    if bet_type == "win_place":
        for tr in soup.select("table.tanpuku tbody tr"):
            horse = _selection(tr.select_one("td.num").get_text(" ", strip=True)) if tr.select_one("td.num") else None
            win = _number(tr.select_one("td.odds_tan").get_text(" ", strip=True)) if tr.select_one("td.odds_tan") else None
            place_node = tr.select_one("td.odds_fuku")
            place_min = _number(place_node.select_one(".min").get_text(strip=True)) if place_node and place_node.select_one(".min") else None
            place_max = _number(place_node.select_one(".max").get_text(strip=True)) if place_node and place_node.select_one(".max") else place_min
            for item in (
                _row(race_id, race_date, "win", [horse], win, win, path),
                _row(race_id, race_date, "place", [horse], place_min, place_max, path),
            ):
                if item:
                    rows.append(item)
    else:
        # 内部券種名とJRA HTMLのテーブルクラス名の対応。
        table_classes = {
            "bracket_quinella": "waku", "quinella": "umaren",
            "wide": "wide", "exacta": "umatan", "trio": "fuku3",
            "trifecta": "tan3",
        }
        # 現在解析する券種に対応した表のCSSクラス。未対応なら表を選択しない。
        table_class = table_classes.get(bet_type)
        for table in soup.select(f"table.{table_class}") if table_class else []:
            if bet_type == "trifecta":
                parent = table.find_parent("li")
                prefix = [
                    _selection(node.get_text(" ", strip=True))
                    for node in (parent.select("div.p_line > div.inner > div.num")[:2] if parent else [])
                ]
            else:
                caption = table.find("caption")
                caption_text = caption.get_text("-", strip=True) if caption else ""
                if bet_type == "bracket_quinella" and caption:
                    caption_text = " ".join(
                        [
                            caption_text,
                            " ".join(caption.get("class", [])),
                            caption.find("img").get("alt", "")
                            if caption.find("img")
                            else "",
                        ]
                    )
                prefix = [int(value) for value in re.findall(r"\d+", caption_text)]
                # 枠連は馬番ではなく枠番で精算する。同枠だけの返還を全枠組合せへ広げない。
                if bet_type == "bracket_quinella":
                    prefix = prefix[:1]
            for tr in table.select("tbody tr"):
                th = tr.find("th")
                td = tr.find("td")
                final = _selection(th.get_text(" ", strip=True)) if th else None
                if td is None:
                    continue
                minimum_node = td.select_one(".min")
                maximum_node = td.select_one(".max")
                minimum = _number(minimum_node.get_text(strip=True)) if minimum_node else _number(td.get_text(" ", strip=True))
                maximum = _number(maximum_node.get_text(strip=True)) if maximum_node else minimum
                item = _row(
                    race_id, race_date, bet_type, [*prefix, final],
                    minimum, maximum, path,
                )
                if item:
                    rows.append(item)
    return pd.DataFrame(rows, columns=ODDS_COLUMNS)


# 対象ディレクトリのHTMLを走査し、解析できた券種の行をまとめてDB登録用の表へ返す。
def parse_odds_directory(
    directory: Path,
    race_date: date,
    race_ids: list[str] | None = None,
) -> pd.DataFrame:
    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    paths = (
        [path for race_id in race_ids for path in directory.glob(f"{race_id}_*.html")]
        if race_ids
        else list(directory.glob("*.html"))
    )
    frames = [parse_odds_html(path, race_date) for path in sorted(set(paths))]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ODDS_COLUMNS)
