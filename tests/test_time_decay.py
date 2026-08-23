import math

import pandas as pd

from src.public_api import (
    calculate_elapsed_days,
    decay_weight,
    decay_weights,
    effective_count,
    weighted_mean,
    weighted_sum,
)


def test_decay_weight_uses_half_life_definition():
    assert decay_weight(0, 180) == 1.0
    assert decay_weight(180, 180) == 0.5
    assert decay_weight(360, 180) == 0.25


def test_decay_weights_preserve_series_index_and_missing_values():
    elapsed = pd.Series([0, 180, None], index=[10, 20, 30])

    result = decay_weights(elapsed, 180)

    assert result.index.tolist() == [10, 20, 30]
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 0.5
    assert math.isnan(result.iloc[2])


def test_elapsed_days_support_fractional_days():
    result = calculate_elapsed_days(
        ["2026-01-03 12:00:00"],
        ["2026-01-02 00:00:00"],
    )
    assert result.iloc[0] == 1.5


def test_future_history_is_rejected():
    try:
        calculate_elapsed_days(["2026-01-01"], ["2026-01-02"])
    except ValueError as exc:
        assert "Future records" in str(exc)
    else:
        raise AssertionError("Future history was accepted")


def test_invalid_half_life_is_rejected():
    for half_life in (0, -1, float("inf")):
        try:
            decay_weights([1, 2], half_life)
        except ValueError as exc:
            assert "half_life_days" in str(exc)
        else:
            raise AssertionError(f"Invalid half-life was accepted: {half_life}")


def test_weighted_aggregates_ignore_missing_value_pairs():
    values = [80.0, 70.0, None]
    weights = [1.0, 0.5, 0.25]

    assert weighted_sum(values, weights) == 115.0
    assert weighted_mean(values, weights) == 115.0 / 1.5
    assert effective_count(weights) == 1.75


def test_weighted_mean_without_effective_weight_is_missing():
    assert math.isnan(weighted_mean([10.0, 20.0], [0.0, 0.0]))
