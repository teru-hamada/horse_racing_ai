# 過去走の経過日数に応じた指数減衰の重みと、重み付き集計を提供する。
# 半減期の日数が過ぎると重みは半分になる。未来の履歴や不正な半減期を拒否する。
# 欠損の値・重みは組で除外し、情報がない場合のNaNと件数ゼロを使い分ける。
#
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


NumericInput = pd.Series | Iterable[float] | np.ndarray
DateInput = pd.Series | Iterable[object]


# 半減期を浮動小数点へ変換し、有限かつ正であることを保証する。
def _validate_half_life(half_life_days: float) -> float:
    value = float(half_life_days)
    # 半減期ゼロ・負値・無限大・NaNでは減衰の意味が定まらないため拒否する。
    if not math.isfinite(value) or value <= 0:
        raise ValueError("half_life_days must be a finite number greater than zero")
    return value


# 配列や反復可能な入力を数値Seriesへそろえる。既存Seriesの添字は維持する。
def _numeric_series(values: NumericInput, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        result = pd.to_numeric(values, errors="coerce").astype(float)
        result.name = name
        return result
    return pd.Series(pd.to_numeric(pd.Series(values), errors="coerce"), name=name, dtype=float)


# 対象日時との差を日単位で求める。時間差も小数日として残し、未来の履歴は拒否する。
def calculate_elapsed_days(
    target_datetimes: DateInput,
    historical_datetimes: DateInput,
) -> pd.Series:
    """Calculate target minus history in fractional days.

    Negative values are rejected because they indicate that a future result
    would be used to construct a point-in-time feature.
    """

    targets = pd.to_datetime(pd.Series(target_datetimes), errors="coerce")
    history = pd.to_datetime(pd.Series(historical_datetimes), errors="coerce")
    # 日時は同じ位置同士の差を取るので、要素数が違う入力は対応が不明として拒否する。
    if len(targets) != len(history):
        raise ValueError("target_datetimes and historical_datetimes must have equal length")

    elapsed = (targets - history).dt.total_seconds() / 86_400.0
    # 負の経過日数は未来の結果を参照することを意味するため、計算を止める。
    if (elapsed.dropna() < 0).any():
        raise ValueError("Future records cannot be used for time-decayed features")
    elapsed.name = "elapsed_days"
    return elapsed


# 重みは0.5の「経過日数÷半減期」乗。経過ゼロなら1、半減期一回分で0.5となる。
def decay_weight(elapsed_days: float, half_life_days: float) -> float:
    """Return one exponential-decay weight using a configurable half-life."""

    # half_life_days: 観測の重みが半分になるまでの経過日数。
    half_life = _validate_half_life(half_life_days)
    days = float(elapsed_days)
    # 欠損の経過日数は重みも欠損として保持し、観測ゼロ日とは区別する。
    if math.isnan(days):
        return math.nan
    if not math.isfinite(days):
        raise ValueError("elapsed_days must be finite")
    # 対象日より後の観測に重みを付けない。未来情報として明示的に拒否する。
    if days < 0:
        raise ValueError("Future records cannot be used for time-decayed features")
    return float(math.pow(0.5, days / half_life))


# Seriesの添字を維持して重みを計算し、欠損の経過日数は欠損のまま残す。
def decay_weights(
    elapsed_days: NumericInput,
    half_life_days: float,
) -> pd.Series:
    """Vectorized version of :func:`decay_weight`, preserving Series indexes."""

    # half_life_days: 観測の重みが半分になるまでの経過日数。
    half_life = _validate_half_life(half_life_days)
    days = _numeric_series(elapsed_days, "elapsed_days")
    finite = days.dropna().map(math.isfinite)
    if not finite.all():
        raise ValueError("elapsed_days must contain only finite values or missing values")
    # ベクトル入力でも一つでも未来の観測があれば拒否する。
    if (days.dropna() < 0).any():
        raise ValueError("Future records cannot be used for time-decayed features")
    # 経過日数に応じて減衰した観測の重み。加重平均と有効件数の計算に使う。
    weights = np.power(0.5, days / half_life)
    return pd.Series(weights, index=days.index, name="decay_weight", dtype=float)


# 値・重みの長さと重みの範囲を検査し、両方が欠損でない組だけを返す。
def _valid_value_weights(
    values: NumericInput,
    weights: NumericInput,
) -> tuple[pd.Series, pd.Series]:
    numeric_values = _numeric_series(values, "value").reset_index(drop=True)
    numeric_weights = _numeric_series(weights, "weight").reset_index(drop=True)
    # 値と重みは位置で対応するため、長さ不一致を切り捨てなどで補わない。
    if len(numeric_values) != len(numeric_weights):
        raise ValueError("values and weights must have equal length")
    # 負の重みを許すと件数や平均の意味が変わるため、集計前に拒否する。
    if (numeric_weights.dropna() < 0).any():
        raise ValueError("weights cannot be negative")
    if not numeric_weights.dropna().map(math.isfinite).all():
        raise ValueError("weights must contain only finite values or missing values")
    valid = numeric_values.notna() & numeric_weights.notna()
    return numeric_values[valid], numeric_weights[valid]


# 値と重みが両方有効な組だけを使う。有効な組がない場合はゼロではなくNaNを返す。
def weighted_sum(values: NumericInput, weights: NumericInput) -> float:
    """Return the weighted sum, or NaN when no valid value/weight pair exists."""

    valid_values, valid_weights = _valid_value_weights(values, weights)
    # 観測がない場合の加重和を、観測されたゼロと区別するためNaNにする。
    if valid_values.empty:
        return math.nan
    return float((valid_values * valid_weights).sum())


# 有効な組の重み合計で正規化する。分母となる有効重みがなければNaNを返す。
def weighted_mean(values: NumericInput, weights: NumericInput) -> float:
    """Return the normalized weighted mean, ignoring missing pairs."""

    valid_values, valid_weights = _valid_value_weights(values, weights)
    # 有効な重みの合計。加重平均の分母または実効的な観測件数となる。
    total_weight = float(valid_weights.sum())
    # 有効な重みがなければ平均は未定義。ゼロではなくNaNを返す。
    if valid_values.empty or total_weight <= 0:
        return math.nan
    return float((valid_values * valid_weights).sum() / total_weight)


# 減衰した重みの合計を有効サンプル数として返す。単なる行数ではない。
def effective_count(weights: NumericInput) -> float:
    """Return the sum of valid decay weights (the effective sample count)."""

    numeric_weights = _numeric_series(weights, "weight")
    # 負の重みを許すと件数や平均の意味が変わるため、集計前に拒否する。
    if (numeric_weights.dropna() < 0).any():
        raise ValueError("weights cannot be negative")
    if not numeric_weights.dropna().map(math.isfinite).all():
        raise ValueError("weights must contain only finite values or missing values")
    return float(numeric_weights.dropna().sum())
