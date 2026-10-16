# 事前に生成した1走ごとの速度指数を、対象日より前の馬別履歴として集約する。
# 直近3走・5走と時間減衰の統計を作り、同日や対象レース自身の指数を参照しない。
# 欠損指数は集計から外し、有効な履歴がなければ値はNaN・件数はゼロを返す。
#
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..base import FEATURE_ID_COLUMNS, FeatureGenerator
from ..context import FeatureContext


# 保存済みの速度指数を馬別に集約し、直近値と時間減衰統計を生成する。
class RecentSpeedGenerator(FeatureGenerator):
    """Aggregate only speed indexes recorded before each target race date."""

    # 登録・保存に使う内部名。画面ラベルとは別に扱う。
    name = "recent_speed"
    # この生成器または定義の版。計算内容を変更するときの識別に使う。
    version = "1.0.0"
    # transformが返す特徴量列の宣言。識別キーを除いた列集合を出力検証で確認する。
    output_columns = (
        "recent_speed_last",
        "recent_speed_mean_3",
        "recent_speed_mean_5",
        "recent_speed_max_5",
        "recent_speed_decay_mean",
        "recent_speed_std_5",
        "recent_speed_available_count_5",
        "recent_speed_effective_count",
    )

    # 近走速度の集約中に呼ぶ進捗通知と取消確認のコールバックを保持する。
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

    # 日付境界を左側で検索して同日の指数を除外する。直近統計は有効な過去指数、減衰統計は期間内の指数を使う。
    def transform(
        self, targets: pd.DataFrame, history: pd.DataFrame, context: FeatureContext
    ) -> pd.DataFrame:
        # targets: 特徴量を付与する対象行。race_id＋horse_idと並び順を維持する。
        # history: 履歴の入力表。集計対象の日付制限はこの処理の境界条件に従う。
        # context: セット名・版と生成器別の計算パラメーター。
        target_required = {"race_id", "horse_id", "race_date"}
        history_required = {"horse_id", "race_date", "speed_index"}
        if missing := target_required.difference(targets.columns):
            raise ValueError(f"Recent-speed targets are missing: {sorted(missing)}")
        if missing := history_required.difference(history.columns):
            raise ValueError(f"Speed-index history is missing: {sorted(missing)}")
        # 近走速度の半減期と最大参照日数を取り出すための生成器別設定。
        settings = context.parameters.get("recent_speed", {})
        # 減衰平均の半減期（日）。直近3走・5走の単純統計とは別に適用する。
        half_life = float(settings.get("half_life_days", 180.0))
        # 減衰平均へ含める観測の最大経過日数。直近統計の走数制限とは異なる。
        max_days = int(settings.get("max_lookback_days", 1095))
        # 半減期と参照期間が正でなければ、重みと履歴範囲を定義できないため拒否する。
        if half_life <= 0 or max_days <= 0:
            raise ValueError("Recent-speed decay and lookback must be positive")

        # 対象行の開催日を日時型へそろえた値。履歴の境界検索と入力検査に使う。
        target_dates = pd.to_datetime(targets["race_date"], errors="coerce")
        # 対象日の欠損は履歴境界を決められないため、対象行を黙って落とさず拒否する。
        if target_dates.isna().any():
            raise ValueError("Recent-speed target race_date is invalid")
        # 馬ID・開催日・速度指数だけに絞り、日付と指数を正規化した集約元の表。
        performance = history[["horse_id", "race_date", "speed_index"]].copy()
        performance["race_date"] = pd.to_datetime(
            performance["race_date"], errors="coerce"
        )
        performance["speed_index"] = pd.to_numeric(
            performance["speed_index"], errors="coerce"
        )
        performance = performance.dropna(subset=["horse_id", "race_date"])

        # 馬IDごとに時系列配列を保持する索引。日付の境界検索に利用する。
        histories: dict[object, tuple[np.ndarray, np.ndarray]] = {}
        groups = performance.groupby("horse_id", sort=False)
        group_count = max(groups.ngroups, 1)
        for position, (horse_id, group) in enumerate(groups):
            # 250単位ごとに取消要求を確認する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._check_cancelled and position % 250 == 0:
                self._check_cancelled()
            # 250単位ごとに進捗を通知する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._progress and position % 250 == 0:
                self._progress(0.2 * position / group_count)
            ordered = group.sort_values("race_date", kind="stable")
            histories[horse_id] = (
                ordered["race_date"].to_numpy(dtype="datetime64[ns]"),
                ordered["speed_index"].to_numpy(dtype=float),
            )

        rows: list[dict[str, object]] = []
        target_count = max(len(targets), 1)
        for position, target in enumerate(targets.itertuples(index=False)):
            # 1000単位ごとに取消要求を確認する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._check_cancelled and position % 1000 == 0:
                self._check_cancelled()
            # 1000単位ごとに進捗を通知する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._progress and position % 1000 == 0:
                self._progress(0.2 + 0.8 * position / target_count)
            arrays = histories.get(target.horse_id)
            target_date = np.datetime64(target_dates.iloc[position].to_datetime64(), "ns")
            row: dict[str, object] = {
                "race_id": target.race_id,
                "horse_id": target.horse_id,
            }
            # 履歴を持たない馬。結果を捏造せず、集計値の欠損と件数ゼロを返す。
            if arrays is None:
                row.update(self._empty())
                rows.append(row)
                continue
            dates, values = arrays
            # 左境界を選び、同日時の行を一件も含めない。対象走の指数による答えの混入を防ぐ。
            end = int(np.searchsorted(dates, target_date, side="left"))
            # 対象日時より前の履歴日付。対象走と同日の行は含めない。
            prior_dates = dates[:end]
            # 対象日時より前に記録された指数。有効な指数へ絞って直近集計に使う。
            prior_values = values[:end]
            # 過去指数がNaNでない位置のマスク。日付と値を同じ位置で除外する。
            valid_prior = ~np.isnan(prior_values)
            prior_dates = prior_dates[valid_prior]
            prior_values = prior_values[valid_prior]
            # 欠損指数を除いた結果が空なら、集計対象なしの値を返す。
            if not len(prior_values):
                row.update(self._empty())
                rows.append(row)
                continue

            elapsed = (target_date - prior_dates) / np.timedelta64(1, "D")
            # 最大参照日数は減衰平均に適用する。直近3走・5走は有効な過去指数の末尾から取る。
            decay_mask = elapsed <= max_days
            # 減衰集計で使う期間内の指数。
            decay_values = prior_values[decay_mask]
            # 減衰集計対象の各指数についての経過日数。
            decay_elapsed = elapsed[decay_mask]
            # 経過日数に応じて減衰した観測の重み。加重平均と有効件数の計算に使う。
            weights = np.power(0.5, decay_elapsed / half_life)
            # 有効な重みの合計。加重平均の分母または実効的な観測件数となる。
            total_weight = float(weights.sum())
            # 欠損を除いた過去の速度指数のうち直近5件以下。
            last_five = prior_values[-5:]
            row.update({
                "recent_speed_last": float(prior_values[-1]),
                "recent_speed_mean_3": float(prior_values[-3:].mean()),
                "recent_speed_mean_5": float(last_five.mean()),
                "recent_speed_max_5": float(last_five.max()),
                "recent_speed_decay_mean": (
                    float(np.dot(decay_values, weights) / total_weight)
                    if total_weight > 0 else np.nan
                ),
                "recent_speed_std_5": float(last_five.std(ddof=0)),
                "recent_speed_available_count_5": float(len(last_five)),
                "recent_speed_effective_count": total_weight,
            })
            rows.append(row)
        # 進捗通知先が指定された場合だけ、計算の終了を1.0で通知する。
        if self._progress:
            self._progress(1.0)
        return pd.DataFrame(rows, columns=FEATURE_ID_COLUMNS + self.output_columns)

    # 未観測の能力をゼロと誤認させないよう統計値はNaNとし、利用可能件数だけゼロにする。
    def _empty(self) -> dict[str, float]:
        values = {column: np.nan for column in self.output_columns}
        values["recent_speed_available_count_5"] = 0.0
        values["recent_speed_effective_count"] = 0.0
        return values
