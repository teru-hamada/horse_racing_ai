from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


NumericInput = pd.Series | Iterable[float] | np.ndarray
DateInput = pd.Series | Iterable[object]


def _validate_half_life(half_life_days: float) -> float:
    value = float(half_life_days)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("half_life_days must be a finite number greater than zero")
    return value


def _numeric_series(values: NumericInput, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        result = pd.to_numeric(values, errors="coerce").astype(float)
        result.name = name
        return result
    return pd.Series(pd.to_numeric(pd.Series(values), errors="coerce"), name=name, dtype=float)


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
    if len(targets) != len(history):
        raise ValueError("target_datetimes and historical_datetimes must have equal length")

    elapsed = (targets - history).dt.total_seconds() / 86_400.0
    if (elapsed.dropna() < 0).any():
        raise ValueError("Future records cannot be used for time-decayed features")
    elapsed.name = "elapsed_days"
    return elapsed


def decay_weight(elapsed_days: float, half_life_days: float) -> float:
    """Return one exponential-decay weight using a configurable half-life."""

    half_life = _validate_half_life(half_life_days)
    days = float(elapsed_days)
    if math.isnan(days):
        return math.nan
    if not math.isfinite(days):
        raise ValueError("elapsed_days must be finite")
    if days < 0:
        raise ValueError("Future records cannot be used for time-decayed features")
    return float(math.pow(0.5, days / half_life))


def decay_weights(
    elapsed_days: NumericInput,
    half_life_days: float,
) -> pd.Series:
    """Vectorized version of :func:`decay_weight`, preserving Series indexes."""

    half_life = _validate_half_life(half_life_days)
    days = _numeric_series(elapsed_days, "elapsed_days")
    finite = days.dropna().map(math.isfinite)
    if not finite.all():
        raise ValueError("elapsed_days must contain only finite values or missing values")
    if (days.dropna() < 0).any():
        raise ValueError("Future records cannot be used for time-decayed features")
    weights = np.power(0.5, days / half_life)
    return pd.Series(weights, index=days.index, name="decay_weight", dtype=float)


def _valid_value_weights(
    values: NumericInput,
    weights: NumericInput,
) -> tuple[pd.Series, pd.Series]:
    numeric_values = _numeric_series(values, "value").reset_index(drop=True)
    numeric_weights = _numeric_series(weights, "weight").reset_index(drop=True)
    if len(numeric_values) != len(numeric_weights):
        raise ValueError("values and weights must have equal length")
    if (numeric_weights.dropna() < 0).any():
        raise ValueError("weights cannot be negative")
    if not numeric_weights.dropna().map(math.isfinite).all():
        raise ValueError("weights must contain only finite values or missing values")
    valid = numeric_values.notna() & numeric_weights.notna()
    return numeric_values[valid], numeric_weights[valid]


def weighted_sum(values: NumericInput, weights: NumericInput) -> float:
    """Return the weighted sum, or NaN when no valid value/weight pair exists."""

    valid_values, valid_weights = _valid_value_weights(values, weights)
    if valid_values.empty:
        return math.nan
    return float((valid_values * valid_weights).sum())


def weighted_mean(values: NumericInput, weights: NumericInput) -> float:
    """Return the normalized weighted mean, ignoring missing pairs."""

    valid_values, valid_weights = _valid_value_weights(values, weights)
    total_weight = float(valid_weights.sum())
    if valid_values.empty or total_weight <= 0:
        return math.nan
    return float((valid_values * valid_weights).sum() / total_weight)


def effective_count(weights: NumericInput) -> float:
    """Return the sum of valid decay weights (the effective sample count)."""

    numeric_weights = _numeric_series(weights, "weight")
    if (numeric_weights.dropna() < 0).any():
        raise ValueError("weights cannot be negative")
    if not numeric_weights.dropna().map(math.isfinite).all():
        raise ValueError("weights must contain only finite values or missing values")
    return float(numeric_weights.dropna().sum())
