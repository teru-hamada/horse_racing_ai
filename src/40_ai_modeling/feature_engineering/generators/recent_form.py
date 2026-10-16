# 過去の着順・上がり3Fなどから、対象日時点の近走成績を集約する。
# 同日以降の結果を履歴から除き、直近走の集計と時間減衰による集計を作る。
# 初出走など履歴がない馬は、成績を推測して埋めずNaNと利用可能件数ゼロで表す。
#
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ..base import FEATURE_ID_COLUMNS, FeatureGenerator
from ..context import FeatureContext


# 特徴量付与対象に必要なキーと開催日。
REQUIRED_TARGET_COLUMNS = ("race_id", "horse_id", "race_date")
# 近走集計の元履歴に必要な結果列。市場情報や馬体重は含めない。
REQUIRED_HISTORY_COLUMNS = (
    "race_id",
    "horse_id",
    "race_date",
    "finish_position",
    "last_3f_time",
)


# 必要列の欠落をまとめて検出し、入力表の種類が分かるエラーを返す。
def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


# 着順と上がり3Fを数値化し、重複を整理してレース内の相対指標を用意する。
def prepare_historical_performances(history: pd.DataFrame) -> pd.DataFrame:
    """Create result-only, race-relative values used by recent-form aggregation."""

    # history: 履歴の入力表。集計対象の日付制限はこの処理の境界条件に従う。
    _require_columns(history, REQUIRED_HISTORY_COLUMNS, "History")
    selected_columns = list(REQUIRED_HISTORY_COLUMNS)
    if "collected_at" in history.columns:
        selected_columns.append("collected_at")
    # 結果列を数値化し、重複整理とレース内の相対指標計算を済ませた全馬の履歴。
    prepared = history.loc[:, selected_columns].copy()
    prepared["race_date"] = pd.to_datetime(prepared["race_date"], errors="coerce")
    prepared["finish_position"] = pd.to_numeric(
        prepared["finish_position"], errors="coerce"
    )
    prepared["last_3f_time"] = pd.to_numeric(prepared["last_3f_time"], errors="coerce")
    if prepared[["race_id", "horse_id"]].isna().any().any():
        raise ValueError("History race_id and horse_id cannot be missing")
    duplicate_keys = prepared.duplicated(["race_id", "horse_id"], keep=False)
    # 重複出走を二重集計しないよう、最新の収集値を選ぶか矛盾を拒否する。
    if duplicate_keys.any():
        if "collected_at" in prepared.columns:
            prepared["collected_at"] = pd.to_datetime(
                prepared["collected_at"], errors="coerce"
            )
            prepared = prepared.sort_values(
                ["race_id", "horse_id", "collected_at"],
                kind="stable",
                na_position="first",
            ).drop_duplicates(["race_id", "horse_id"], keep="last")
        else:
            prepared = prepared.drop_duplicates()
            if prepared.duplicated(["race_id", "horse_id"]).any():
                raise ValueError(
                    "History contains conflicting duplicate race/horse rows "
                    "without collected_at"
                )
    prepared = prepared.drop(columns=["collected_at"], errors="ignore")

    prepared["field_size"] = prepared.groupby("race_id")["horse_id"].transform("size")
    valid_finish = prepared["finish_position"].gt(0)
    denominator = prepared["field_size"] - 1
    prepared["relative_finish"] = np.where(
        valid_finish & denominator.gt(0),
        (prepared["finish_position"] - 1) / denominator,
        np.nan,
    )
    prepared["is_win"] = prepared["finish_position"].eq(1).where(valid_finish)
    prepared["is_top3"] = prepared["finish_position"].le(3).where(valid_finish)

    valid_last3f_count = prepared.groupby("race_id")["last_3f_time"].transform("count")
    last3f_rank = prepared.groupby("race_id")["last_3f_time"].rank(
        method="average", ascending=True, na_option="keep"
    )
    prepared["last3f_rank_rate"] = np.where(
        valid_last3f_count.gt(1),
        (last3f_rank - 1) / (valid_last3f_count - 1),
        np.nan,
    )
    return prepared.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)


# コンテキストから近走成績の減衰条件を読み取り、計算可能な値か検査する。
def _settings(parameters: Mapping[str, Any]) -> tuple[float, int]:
    configured = parameters.get("recent_form", {})
    if not isinstance(configured, Mapping):
        raise ValueError("recent_form parameters must be a mapping")
    half_life_days = float(configured.get("half_life_days", 180.0))
    max_lookback_days = int(configured.get("max_lookback_days", 1095))
    if max_lookback_days <= 0:
        raise ValueError("max_lookback_days must be greater than zero")
    return half_life_days, max_lookback_days


# 欠損を除いた平均を計算する。有効値がなければNaNとする。
def _nanmean(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(valid.mean()) if len(valid) else np.nan


# 欠損を除いた標準偏差を計算し、有効な観測がないケースを扱う。
def _nanstd(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(valid.std(ddof=0)) if len(valid) else np.nan


# 値と重みの有効な組に絞り、重み合計で割った平均を計算する。
def _weighted_array_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = ~np.isnan(values) & ~np.isnan(weights)
    # 有効な重みの合計。加重平均の分母または実効的な観測件数となる。
    total_weight = float(weights[valid].sum())
    if not valid.any() or total_weight <= 0:
        return np.nan
    return float(np.dot(values[valid], weights[valid]) / total_weight)


# 着順など結果由来の履歴を、対象日より前だけに限定して集計する生成器。
class RecentFormGenerator(FeatureGenerator):
    """Aggregate point-in-time form without odds, body weight, or speed indexes."""

    # 登録・保存に使う内部名。画面ラベルとは別に扱う。
    name = "recent_form"
    # この生成器または定義の版。計算内容を変更するときの識別に使う。
    version = "1.0.0"
    # transformが返す特徴量列の宣言。識別キーを除いた列集合を出力検証で確認する。
    output_columns = (
        "recent_starts_total",
        "recent_days_since_last",
        "recent_finish_last",
        "recent_finish_mean_3",
        "recent_finish_mean_5",
        "recent_relative_finish_last",
        "recent_relative_finish_mean_3",
        "recent_relative_finish_mean_5",
        "recent_relative_finish_decay_mean",
        "recent_win_rate_5",
        "recent_top3_rate_5",
        "recent_win_rate_decay",
        "recent_top3_rate_decay",
        "recent_last3f_rank_rate_last",
        "recent_last3f_rank_rate_mean_3",
        "recent_last3f_rank_rate_decay_mean",
        "recent_relative_finish_std_5",
        "recent_finish_available_count_5",
        "recent_effective_starts",
    )

    # 進捗通知と取消確認の関数を保持する。履歴そのものはtransformへ渡す。
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

    # 馬別の時系列配列を準備し、各対象行を_aggregateへ渡す。そこで同日以降を除外する。
    def transform(
        self,
        targets: pd.DataFrame,
        history: pd.DataFrame,
        context: FeatureContext,
    ) -> pd.DataFrame:
        # targets: 特徴量を付与する対象行。race_id＋horse_idと並び順を維持する。
        # history: 履歴の入力表。集計対象の日付制限はこの処理の境界条件に従う。
        # context: セット名・版と生成器別の計算パラメーター。
        _require_columns(targets, REQUIRED_TARGET_COLUMNS, "Targets")
        half_life_days, max_lookback_days = _settings(context.parameters)
        prepared = prepare_historical_performances(history)
        # Prediction targets contain only one race. Race-relative values must be
        # calculated with the full history first, but histories for horses that
        # are not in the target race do not need to be indexed afterward.
        target_horses = set(targets["horse_id"].dropna().tolist())
        prepared = prepared[prepared["horse_id"].isin(target_horses)]
        # 馬IDから時系列の結果配列を取り出す索引。対象行ごとの履歴再走査を避ける。
        histories_by_horse = {}
        grouped = prepared.groupby("horse_id", sort=False)
        group_count = max(grouped.ngroups, 1)
        for group_position, (horse_id, group) in enumerate(grouped):
            # 250単位ごとに取消要求を確認する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._check_cancelled and group_position % 250 == 0:
                self._check_cancelled()
            # 250単位ごとに進捗を通知する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._progress and group_position % 250 == 0:
                self._progress(0.25 * group_position / group_count)
            ordered = group.sort_values(["race_date", "race_id"], kind="stable")
            histories_by_horse[horse_id] = {
                "dates": ordered["race_date"].to_numpy(dtype="datetime64[ns]"),
                "finish": ordered["finish_position"].to_numpy(dtype=float),
                "relative": ordered["relative_finish"].to_numpy(dtype=float),
                "win": ordered["is_win"].to_numpy(dtype=float, na_value=np.nan),
                "top3": ordered["is_top3"].to_numpy(dtype=float, na_value=np.nan),
                "last3f": ordered["last3f_rank_rate"].to_numpy(dtype=float),
            }
        # 対象行の開催日を日時型へそろえた値。履歴の境界検索と入力検査に使う。
        target_dates = pd.to_datetime(targets["race_date"], errors="coerce")
        # 対象日の欠損は履歴境界を決められないため、対象行を黙って落とさず拒否する。
        if target_dates.isna().any():
            raise ValueError("Target race_date cannot be missing or invalid")

        rows: list[dict[str, object]] = []
        target_count = max(len(targets), 1)
        for position, target in enumerate(targets.itertuples(index=False)):
            # 1000単位ごとに取消要求を確認する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._check_cancelled and position % 1000 == 0:
                self._check_cancelled()
            # 1000単位ごとに進捗を通知する。各行で呼ぶ負荷を抑えつつ長時間処理の反応を保つ。
            if self._progress and position % 1000 == 0:
                self._progress(0.25 + 0.75 * position / target_count)
            target_date = target_dates.iloc[position]
            arrays = histories_by_horse.get(target.horse_id)
            rows.append(
                self._aggregate(
                    target.race_id,
                    target.horse_id,
                    target_date,
                    arrays,
                    half_life_days,
                    max_lookback_days,
                )
            )
        # 進捗通知先が指定された場合だけ、計算の終了を1.0で通知する。
        if self._progress:
            self._progress(1.0)
        return pd.DataFrame(rows, columns=FEATURE_ID_COLUMNS + self.output_columns)

    # 直近の走数ごとの成績と時間減衰集計を作り、履歴なしは件数ゼロ・統計NaNで表現する。
    def _aggregate(
        self,
        race_id: object,
        horse_id: object,
        target_date: pd.Timestamp,
        arrays: dict[str, np.ndarray] | None,
        half_life_days: float,
        max_lookback_days: int,
    ) -> dict[str, object]:
        # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
        # half_life_days: 観測の重みが半分になるまでの経過日数。
        # max_lookback_days: 減衰集計へ参照する履歴の上限日数。
        result: dict[str, object] = {
            "race_id": race_id,
            "horse_id": horse_id,
        }
        # 履歴を持たない馬。結果を捏造せず、集計値の欠損と件数ゼロを返す。
        if arrays is None:
            result["recent_starts_total"] = 0.0
            result.update({column: np.nan for column in self.output_columns[1:]})
            result["recent_finish_available_count_5"] = 0.0
            result["recent_effective_starts"] = 0.0
            return result

        # 対象日をNumPyの日時へそろえた境界値。対象日と同日の行を検索範囲から除く。
        target_value = np.datetime64(target_date.to_datetime64(), "ns")
        # 対象日より厳密に前の出走行数。searchsortedの左境界から求める。
        starts = int(np.searchsorted(arrays["dates"], target_value, side="left"))
        result["recent_starts_total"] = float(starts)
        # 馬の履歴自体はあっても、対象日より前の出走がなければ初出走と同じ扱いにする。
        if starts == 0:
            result.update({column: np.nan for column in self.output_columns[1:]})
            result["recent_finish_available_count_5"] = 0.0
            result["recent_effective_starts"] = 0.0
            return result

        dates = arrays["dates"][:starts]
        finish = arrays["finish"][:starts]
        # 対象日以前のレース内相対着順。出走頭数の違いを考慮した集計に使う。
        relative = arrays["relative"][:starts]
        win = arrays["win"][:starts]
        top3 = arrays["top3"][:starts]
        # 対象日以前の上がり3Fのレース内相対順位。
        last3f = arrays["last3f"][:starts]
        # 対象日から過去走までの経過日数。減衰の重みと参照期間の判定に使う。
        elapsed_days = (target_value - dates) / np.timedelta64(1, "D")
        # 減衰集計の最大参照日数内にある観測だけを選ぶマスク。
        decay_mask = elapsed_days <= max_lookback_days
        # 経過日数に応じて減衰した観測の重み。加重平均と有効件数の計算に使う。
        weights = np.power(0.5, elapsed_days[decay_mask] / half_life_days)

        result.update({
            "recent_days_since_last": float(elapsed_days[-1]),
            "recent_finish_last": finish[-1],
            "recent_finish_mean_3": _nanmean(finish[-3:]),
            "recent_finish_mean_5": _nanmean(finish[-5:]),
            "recent_relative_finish_last": relative[-1],
            "recent_relative_finish_mean_3": _nanmean(relative[-3:]),
            "recent_relative_finish_mean_5": _nanmean(relative[-5:]),
            "recent_relative_finish_decay_mean": _weighted_array_mean(
                relative[decay_mask], weights
            ),
            "recent_win_rate_5": _nanmean(win[-5:]),
            "recent_top3_rate_5": _nanmean(top3[-5:]),
            "recent_win_rate_decay": _weighted_array_mean(win[decay_mask], weights),
            "recent_top3_rate_decay": _weighted_array_mean(top3[decay_mask], weights),
            "recent_last3f_rank_rate_last": last3f[-1],
            "recent_last3f_rank_rate_mean_3": _nanmean(last3f[-3:]),
            "recent_last3f_rank_rate_decay_mean": _weighted_array_mean(
                last3f[decay_mask], weights
            ),
            "recent_relative_finish_std_5": _nanstd(relative[-5:]),
            "recent_finish_available_count_5": float(np.count_nonzero(~np.isnan(finish[-5:]))),
            "recent_effective_starts": float(weights.sum()),
        })
        return result
