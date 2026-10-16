# 出馬表で分かる条件とレース内の相対条件を出走馬ごとに生成する。
# 馬番・枠番・斤量・距離などを使い、前走との距離差は対象日より前の出走だけから計算する。
# オッズ・人気・馬体重・当日の天気はこの生成器の出力に含めない。
#
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..base import FEATURE_ID_COLUMNS, FeatureGenerator
from ..context import FeatureContext


# 出馬表条件として取り出す列の許可リスト。結果列を流用しない。
ENTRY_COLUMNS = (
    "race_id", "horse_id", "race_date", "course_name", "race_number",
    "surface", "distance", "track_condition", "horse_number", "frame_number",
    "sex", "age", "carried_weight",
)


# 出馬表時点で利用できる列だけを取り出し、収集日時があれば重複キーの最新版を採用する。
def prepare_race_entries(records: pd.DataFrame) -> pd.DataFrame:
    """Normalize fields known when the race entry table is available."""

    missing = set(ENTRY_COLUMNS).difference(records.columns)
    # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
    if missing:
        raise ValueError(f"Race entries are missing columns: {sorted(missing)}")
    columns = list(ENTRY_COLUMNS)
    if "collected_at" in records:
        columns.append("collected_at")
    # 対象列に限定し日付・数値・重複を整理した出馬表履歴。
    entries = records.loc[:, columns].copy()
    entries["race_date"] = pd.to_datetime(entries["race_date"], errors="coerce")
    if entries[["race_id", "horse_id", "race_date"]].isna().any().any():
        raise ValueError("Race-entry keys and race_date cannot be missing")
    for column in (
        "race_number", "distance", "horse_number", "frame_number", "age",
        "carried_weight",
    ):
        entries[column] = pd.to_numeric(entries[column], errors="coerce")
    if entries.duplicated(["race_id", "horse_id"]).any():
        # 重複行のどちらが新しいか判定できないため、任意に一件を選ばず拒否する。
        if "collected_at" not in entries:
            raise ValueError("Race entries contain duplicate race_id + horse_id keys")
        entries["collected_at"] = pd.to_datetime(entries["collected_at"], errors="coerce")
        entries = entries.sort_values(
            ["race_id", "horse_id", "collected_at"], kind="stable", na_position="first"
        ).drop_duplicates(["race_id", "horse_id"], keep="last")
    return entries.drop(columns="collected_at", errors="ignore").reset_index(drop=True)


# 当該出馬表の条件とレース内の相対値を生成する。前走距離だけは過去履歴から参照する。
class RaceEntryGenerator(FeatureGenerator):
    """Create entry-table features plus the supplied track condition."""

    # 登録・保存に使う内部名。画面ラベルとは別に扱う。
    name = "race_entry"
    # この生成器または定義の版。計算内容を変更するときの識別に使う。
    version = "1.1.0"
    # transformが返す特徴量列の宣言。識別キーを除いた列集合を出力検証で確認する。
    output_columns = (
        "race_entry_course",
        "race_entry_race_number",
        "race_entry_surface",
        "race_entry_track_condition",
        "race_entry_distance",
        "race_entry_age",
        "race_entry_sex",
        "race_entry_carried_weight",
        "race_entry_frame_number",
        "race_entry_horse_number",
        "race_entry_field_size",
        "race_entry_frame_position_ratio",
        "race_entry_horse_number_ratio",
        "race_entry_carried_weight_diff_field_mean",
        "race_entry_distance_change_from_last",
    )

    # 出馬表特徴量の計算ループから呼ぶ進捗通知と取消確認を保持する。
    def __init__(
        self,
        progress: Callable[[float], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        # progress: 任意の進捗コールバック。値は通常0～1の完了率。
        # check_cancelled: 取消時に例外を送出する任意の関数。長い計算の途中から呼び出す。
        # 処理側から進捗率を通知する任意のコールバック。
        self._progress = progress
        # 長いループや保存前に呼び、取消要求があれば例外で抜けるための関数。
        self._check_cancelled = check_cancelled

    # 対象馬の当該出馬表とレース内の相対値を生成し、前走距離だけは対象日より前の履歴から探す。
    def transform(
        self, targets: pd.DataFrame, history: pd.DataFrame, context: FeatureContext
    ) -> pd.DataFrame:
        # targets: 特徴量を付与する対象行。race_id＋horse_idと並び順を維持する。
        # history: 履歴の入力表。集計対象の日付制限はこの処理の境界条件に従う。
        # context: セット名・版と生成器別の計算パラメーター。
        entries = prepare_race_entries(history)
        # 今回計算するrace_id＋horse_idのキー。無関係な出走結果を出力しない。
        target_keys = targets.loc[:, ["race_id", "horse_id"]].copy()
        # 一頭一行の対象キーが重複していれば、特徴量生成の前提違反として拒否する。
        if target_keys.duplicated().any():
            raise ValueError("Race-entry targets contain duplicate keys")
        # 今回の対象キーに一致した出馬表。レース内の頭数と平均斤量はこの表から求める。
        current = target_keys.merge(
            entries, on=["race_id", "horse_id"], how="left", validate="one_to_one"
        )
        # 対象キーに対応する出馬表行がなければ、欠落したまま相対値を計算しない。
        if current["race_date"].isna().any():
            raise ValueError("Some race-entry targets do not exist in history")

        # 対象出馬表をレース単位でまとめたもの。頭数や平均斤量を馬ごとに展開する。
        race_groups = current.groupby("race_id", sort=False)
        current["_field_size"] = race_groups["horse_id"].transform("size")
        current["_mean_weight"] = race_groups["carried_weight"].transform("mean")

        # 馬IDごとに時系列配列を保持する索引。日付の境界検索に利用する。
        histories: dict[object, tuple[np.ndarray, np.ndarray]] = {}
        for horse_id, group in entries.groupby("horse_id", sort=False):
            ordered = group.sort_values(["race_date", "race_id"], kind="stable")
            histories[horse_id] = (
                ordered["race_date"].to_numpy(dtype="datetime64[ns]"),
                ordered["distance"].to_numpy(dtype=float),
            )

        # 対象馬ごとの前走距離を格納する配列。過去走がなければNaNを維持する。
        previous_distances = np.full(len(current), np.nan)
        total = max(len(current), 1)
        for position, row in enumerate(current.itertuples(index=False)):
            # 1000単位ごとに取消要求を確認する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._check_cancelled and position % 1000 == 0:
                self._check_cancelled()
            # 1000単位ごとに進捗を通知する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._progress and position % 1000 == 0:
                self._progress(position / total)
            dates, distances = histories[row.horse_id]
            target_date = np.datetime64(row.race_date.to_datetime64(), "ns")
            # 当日を含まない前走候補の終端。ゼロなら対象日より前の距離履歴はない。
            end = int(np.searchsorted(dates, target_date, side="left"))
            # 対象日より前の履歴がある場合だけ前走距離を探す。初出走の距離差はNaNのまま残す。
            if end:
                valid = distances[:end][~np.isnan(distances[:end])]
                if len(valid):
                    previous_distances[position] = valid[-1]

        result = pd.DataFrame({
            "race_id": current["race_id"],
            "horse_id": current["horse_id"],
            "race_entry_course": current["course_name"].astype("string"),
            "race_entry_race_number": current["race_number"],
            "race_entry_surface": current["surface"].astype("string"),
            "race_entry_track_condition": current["track_condition"].astype("string"),
            "race_entry_distance": current["distance"],
            "race_entry_age": current["age"],
            "race_entry_sex": current["sex"].astype("string"),
            "race_entry_carried_weight": current["carried_weight"],
            "race_entry_frame_number": current["frame_number"],
            "race_entry_horse_number": current["horse_number"],
            "race_entry_field_size": current["_field_size"].astype(float),
            "race_entry_frame_position_ratio": np.where(
                current["_field_size"].gt(0),
                current["frame_number"] / current["_field_size"],
                np.nan,
            ),
            "race_entry_horse_number_ratio": np.where(
                current["_field_size"].gt(0),
                current["horse_number"] / current["_field_size"],
                np.nan,
            ),
            "race_entry_carried_weight_diff_field_mean": (
                current["carried_weight"] - current["_mean_weight"]
            ),
            "race_entry_distance_change_from_last": (
                current["distance"].to_numpy(dtype=float) - previous_distances
            ),
        })
        # 進捗通知先が指定された場合だけ、計算の終了を1.0で通知する。
        if self._progress:
            self._progress(1.0)
        return result.loc[:, FEATURE_ID_COLUMNS + self.output_columns]
