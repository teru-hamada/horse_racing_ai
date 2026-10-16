# 完了した各出走の走破時間を、過去の標準タイムと斤量から速度指数へ変換する。
# 標準は開催日ごとに計算した後で更新し、同じ日の別レースの勝ち時計を混入させない。
# 条件が一致する過去走が不足すると参照条件を段階的に広げ、指数は生値と0～200への制限後の値を残す。
#
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..base import FEATURE_ID_COLUMNS, FeatureGenerator
from ..context import FeatureContext


# このモジュールの正規化・計算に欠かせない入力列。
REQUIRED_COLUMNS = (
    "race_id", "horse_id", "race_date", "course_name", "surface", "distance",
    "track_condition", "carried_weight", "finish_position", "time_seconds",
)


# 標準タイムと指数の計算に必要な列を正規化し、収集日時で重複する出走結果を整理する。
def prepare_speed_performances(records: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(records.columns)
    # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
    if missing:
        raise ValueError(f"Speed-index records are missing columns: {sorted(missing)}")
    columns = list(REQUIRED_COLUMNS)
    if "collected_at" in records.columns:
        columns.append("collected_at")
    frame = records.loc[:, columns].copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    for column in ("distance", "carried_weight", "finish_position", "time_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["race_id", "horse_id"]].isna().any().any():
        raise ValueError("Speed-index race_id and horse_id cannot be missing")
    duplicates = frame.duplicated(["race_id", "horse_id"], keep=False)
    # 同じ出走の重複があれば、収集日時による選択か完全一致の整理を行う。
    if duplicates.any():
        if "collected_at" not in frame.columns:
            frame = frame.drop_duplicates()
            if frame.duplicated(["race_id", "horse_id"]).any():
                raise ValueError("Conflicting duplicate speed-index rows lack collected_at")
        else:
            frame["collected_at"] = pd.to_datetime(frame["collected_at"], errors="coerce")
            frame = frame.sort_values(
                ["race_id", "horse_id", "collected_at"], kind="stable",
                na_position="first",
            ).drop_duplicates(["race_id", "horse_id"], keep="last")
    return frame.drop(columns=["collected_at"], errors="ignore").sort_values(
        ["race_date", "race_id", "horse_id"], kind="stable"
    ).reset_index(drop=True)


# 一つの参照条件に対応する日付と勝ち時計ペースの並行配列。日付昇順の追加を前提とする。
@dataclass
class _ReferenceSeries:
    # 過去の基準時計が観測された日。日付昇順でpacesと同じ添字に対応する。
    days: list[int]
    # 勝ち時計を1000m当たりの秒数へ換算した参照値。
    paces: list[float]

    # 一つの日付と基準ペースを並行配列へ追加する。呼出側が日付順を保つ。
    def append(self, day: int, pace: float) -> None:
        self.days.append(day)
        self.paces.append(pace)

    # 参照期間内の勝ち時計ペースを半減期で重み付けし、平均・実件数・有効件数を返す。
    def estimate(
        self, target_day: int, half_life_days: float, max_lookback_days: int
    ) -> tuple[float, int, float]:
        # half_life_days: 観測の重みが半分になるまでの経過日数。
        # max_lookback_days: 減衰集計へ参照する履歴の上限日数。
        start = bisect_left(self.days, target_day - max_lookback_days)
        days = np.asarray(self.days[start:], dtype=float)
        values = np.asarray(self.paces[start:], dtype=float)
        # 参照期間内に時計がなければ、標準はNaN・件数はゼロとして返す。
        if not len(values):
            return np.nan, 0, 0.0
        # 経過日数に応じて減衰した観測の重み。加重平均と有効件数の計算に使う。
        weights = np.power(0.5, (target_day - days) / half_life_days)
        total = float(weights.sum())
        return float(np.dot(values, weights) / total), len(values), total


# 過去の勝ち時計を基準に、各出走の時計と斤量を速度指数へ換算する生成器。
class SpeedIndexGenerator(FeatureGenerator):
    """Rate each completed performance using standards known before its date."""

    # 登録・保存に使う内部名。画面ラベルとは別に扱う。
    name = "speed_index"
    # この生成器または定義の版。計算内容を変更するときの識別に使う。
    version = "1.0.0"
    # transformが返す特徴量列の宣言。識別キーを除いた列集合を出力検証で確認する。
    output_columns = (
        "speed_index",
        "speed_index_raw",
        "speed_index_was_clipped",
        "speed_standard_time",
        "speed_time_difference",
        "speed_weight_adjustment",
        "speed_track_adjustment",
        "speed_reference_race_count",
        "speed_reference_effective_count",
        "speed_fallback_level",
    )

    # 日付ごとの速度指数計算に使う進捗通知・取消確認の関数を保持する。
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

    # 日付昇順に指数を計算する。条件別の標準に必要な件数がなければ、より広い条件へフォールバックする。
    def transform(
        self, targets: pd.DataFrame, history: pd.DataFrame, context: FeatureContext
    ) -> pd.DataFrame:
        # targets: 特徴量を付与する対象行。race_id＋horse_idと並び順を維持する。
        # history: 履歴の入力表。集計対象の日付制限はこの処理の境界条件に従う。
        # context: セット名・版と生成器別の計算パラメーター。
        prepared = prepare_speed_performances(history)
        # 速度指数生成器に指定されたパラメーター辞書。未指定の項目には既定値を使う。
        settings = context.parameters.get("speed_index", {})
        # 標準時計の指数減衰の半減期（日）。古い勝ち時計ほど重みを小さくする。
        half_life = float(settings.get("half_life_days", 730.0))
        # 標準時計の推定へ採用する過去の最大日数。
        max_days = int(settings.get("max_lookback_days", 3650))
        # 1600mで1秒の差を指数ポイントへ換算する係数。
        seconds_scale = float(settings.get("seconds_scale_at_1600m", 10.0))
        # 斤量1kgの違いを指数ポイントへ換算する係数。
        weight_points = float(settings.get("weight_points_per_kg", 1.5))
        # 減衰・参照期間・時間差の尺度は正に限定し、意味のない指数を計算しない。
        if half_life <= 0 or max_days <= 0 or seconds_scale <= 0:
            raise ValueError("Speed-index decay, lookback and scale must be positive")

        # 今回計算するrace_id＋horse_idのキー。無関係な出走結果を出力しない。
        target_keys = set(zip(targets["race_id"], targets["horse_id"]))
        # 出走キーごとに計算済み指数を保持し、最後にtargetsと同じ順序へ戻す。
        result_by_key: dict[tuple[object, object], dict[str, object]] = {}
        # 条件キーから過去の基準時計系列を引く辞書。
        references: dict[tuple[object, ...], _ReferenceSeries] = {}

        # 条件キーに対応する参照系列を取得し、初回だけ空の系列を作る。
        def reference(key: tuple[object, ...]) -> _ReferenceSeries:
            return references.setdefault(key, _ReferenceSeries([], []))

        # 有効な開催日がある出走だけに絞った履歴。日付なしの時計を基準系列へ加えない。
        dated = prepared[prepared["race_date"].notna()]
        # 開催日昇順のグループ。翌日の計算へだけ当日の時計を反映できる順序を作る。
        date_groups = list(dated.groupby("race_date", sort=True))
        # 進捗率の分母。空履歴でもゼロ除算しないよう下限を1にする。
        total_dates = max(len(date_groups), 1)
        # 開催日ごとに処理し、当日の基準追加を翌日まで見せない順序を保つ。
        for date_position, (race_date, date_frame) in enumerate(date_groups):
            # 10単位ごとに取消要求を確認する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._check_cancelled and date_position % 10 == 0:
                self._check_cancelled()
            # 10単位ごとに進捗を通知する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._progress and date_position % 10 == 0:
                self._progress(date_position / total_dates)
            # 開催日を整数の日数に変換した値。標準時計系列の期間検索を軽くする。
            target_day = int(race_date.to_datetime64().astype("datetime64[D]").astype(int))
            # 当日の勝ち時計の追加待ち一覧。同日の全レースを採点するまでは基準へ加えない。
            pending_references: list[tuple[tuple[object, ...], float]] = []

            for race_id, race in date_frame.groupby("race_id", sort=False):
                first = race.iloc[0]
                course = first["course_name"]
                surface = first["surface"]
                distance = first["distance"]
                condition = first["track_condition"]
                # コース・馬場種別・正の距離が揃い、標準時計を検索できるかを示す条件。
                valid_race = (
                    pd.notna(course) and pd.notna(surface) and pd.notna(distance)
                    and float(distance) > 0
                )
                # 標準時計・件数・有効件数・採用条件・馬場補正の組。未確定時は欠損とunavailableを保持する。
                estimate = (np.nan, 0, 0.0, "unavailable", 0.0)
                # 基準検索に必要な条件が揃うレースだけ推定する。不足時の標準は欠損のままとする。
                if valid_race:
                    # メートル単位の距離を数値へそろえ、基準ペースと時計の換算に使う。
                    distance_value = float(distance)
                    # 距離を200m幅にまとめた区分。厳密な距離一致で過去件数が足りない場合の参照条件。
                    band = int(distance_value // 200 * 200)
                    # 狭い条件ほど似たレースを参照できるが、必要件数に届かなければ順に条件を広げる。
                    # 各タプルの数値は実レース数の下限で、有効重みの合計とは別の判定基準。
                    keys = [
                        (("condition", course, surface, distance_value, condition), 10, "course_surface_distance_condition"),
                        (("exact", course, surface, distance_value), 20, "course_surface_distance"),
                        (("course_band", course, surface, band), 30, "course_surface_distance_band"),
                        (("surface_band", surface, band), 50, "surface_distance_band"),
                        (("surface", surface), 100, "surface"),
                    ]
                    # 必要件数に最初に届いた参照条件の推定結果。見つからなければNone。
                    chosen = None
                    for key, minimum, label in keys:
                        pace, count, effective = reference(key).estimate(
                            target_day, half_life, max_days
                        )
                        # 最も狭い条件から調べ、実レース数の下限に達した最初の条件を採用する。
                        if count >= minimum:
                            chosen = (pace, count, effective, label)
                            break
                    # 有効な過去基準が見つかったときだけ標準時計へ換算し、推測値で穴埋めしない。
                    if chosen is not None:
                        pace, count, effective, level = chosen
                        # 対象条件の過去の勝ち時計から推定した標準走破時間（秒）。
                        standard_time = pace * distance_value / 1000.0
                        # 馬場状態別の基準と状態を限定しない基準との時間差。
                        track_adjustment = 0.0
                        # 馬場状態まで一致する基準を採用した場合だけ、状態を限定しない基準との差を記録する。
                        if level == "course_surface_distance_condition":
                            base_pace, base_count, _ = reference(
                                ("exact", course, surface, distance_value)
                            ).estimate(target_day, half_life, max_days)
                            # 比較する広い条件の基準にも必要件数がある場合に限り、馬場差を算出する。
                            if base_count >= 20:
                                track_adjustment = (
                                    pace - base_pace
                                ) * distance_value / 1000.0
                        estimate = (
                            standard_time, count, effective, level, track_adjustment
                        )

                standard_time, count, effective, level, track_adjustment = estimate
                for row in race.itertuples(index=False):
                    key = (row.race_id, row.horse_id)
                    # 参照履歴の維持に必要なレースでも、今回の計算対象でない馬は結果として出力しない。
                    if key not in target_keys:
                        continue
                    # 対象出走の走破時間（秒）。標準時計との差を指数へ換算する。
                    actual_time = float(row.time_seconds) if pd.notna(row.time_seconds) else np.nan
                    # 対象馬の斤量（kg）。55kgとの差を指数へ補正する。
                    carried_weight = (
                        float(row.carried_weight) if pd.notna(row.carried_weight) else np.nan
                    )
                    # 標準時計から実走時計を引いた秒差。標準より速いと正になる。
                    difference = (
                        float(standard_time - actual_time)
                        if pd.notna(standard_time) and pd.notna(actual_time) else np.nan
                    )
                    # 斤量55kgを基準にした指数ポイントの加減算。
                    weight_adjustment = (
                        (carried_weight - 55.0) * weight_points
                        if pd.notna(carried_weight) else np.nan
                    )
                    # 標準より速ければ加点し、時間差は1600m換算で尺度をそろえる。
                    # 斤量55kgを基準に補正し、比較に必要な値が欠損していれば指数も欠損とする。
                    raw_score = (
                        100.0
                        + difference * seconds_scale * 1600.0 / float(distance)
                        + weight_adjustment
                        if valid_race and pd.notna(difference) and pd.notna(weight_adjustment)
                        else np.nan
                    )
                    # 範囲を0～200へ制限した最終指数。比較用に生のraw_scoreも別途保存する。
                    score = (
                        float(np.clip(raw_score, 0.0, 200.0))
                        if pd.notna(raw_score) else np.nan
                    )
                    result_by_key[key] = {
                        "race_id": row.race_id,
                        "horse_id": row.horse_id,
                        "speed_index": score,
                        "speed_index_raw": raw_score,
                        "speed_index_was_clipped": (
                            bool(score != raw_score) if pd.notna(raw_score) else False
                        ),
                        "speed_standard_time": standard_time,
                        "speed_time_difference": difference,
                        "speed_weight_adjustment": weight_adjustment,
                        "speed_track_adjustment": track_adjustment,
                        "speed_reference_race_count": float(count),
                        "speed_reference_effective_count": float(effective),
                        "speed_fallback_level": level,
                    }

                # 同レースで1着かつ正の有効時計を持つ行。基準へ追加する勝ち時計を選ぶ。
                winners = race[
                    race["finish_position"].eq(1)
                    & race["time_seconds"].notna()
                    & race["time_seconds"].gt(0)
                ]
                if valid_race and not winners.empty:
                    # 勝ち時計を1000m当たり秒数へ換算した値。異なる距離の参照にも用いる。
                    winning_pace = float(winners.iloc[0]["time_seconds"]) / float(distance) * 1000.0
                    band = int(float(distance) // 200 * 200)
                    for key in (
                        ("condition", course, surface, float(distance), condition),
                        ("exact", course, surface, float(distance)),
                        ("course_band", course, surface, band),
                        ("surface_band", surface, band),
                        ("surface", surface),
                    ):
                        pending_references.append((key, winning_pace))

            # 同日の全レースを評価し終えてから、その日の勝ち時計を参照系列へ追加する。
            # これをレースごとに行うと、同日の別レースの結果が基準へ混入する。
            # Update only after every race on the date has been scored.
            for key, pace in pending_references:
                reference(key).append(target_day, pace)

        rows = []
        for key in zip(targets["race_id"], targets["horse_id"]):
            rows.append(result_by_key.get(key, {
                "race_id": key[0], "horse_id": key[1],
                **{column: np.nan for column in self.output_columns[:-1]},
                "speed_fallback_level": "unavailable",
            }))
        # 進捗通知先が指定された場合だけ、計算の終了を1.0で通知する。
        if self._progress:
            self._progress(1.0)
        return pd.DataFrame(rows, columns=FEATURE_ID_COLUMNS + self.output_columns)
