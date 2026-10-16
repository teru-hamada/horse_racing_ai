# 検証対象：半減期の定義、小数日、Seriesの添字保持、欠損の組を除く重み付き集計を確認する。
# 未来履歴と不正な半減期の拒否、有効な重みがないときのNaNを検証する。
#
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


# 経過ゼロ・半減期一回・二回で重みが1・0.5・0.25となることを確認する。
def test_decay_weight_uses_half_life_definition():
    assert decay_weight(0, 180) == 1.0
    assert decay_weight(180, 180) == 0.5
    assert decay_weight(360, 180) == 0.25


# Seriesの独自添字と欠損位置を維持して、要素ごとの減衰重みを計算する。
def test_decay_weights_preserve_series_index_and_missing_values():
    elapsed = pd.Series([0, 180, None], index=[10, 20, 30])

    result = decay_weights(elapsed, 180)

    assert result.index.tolist() == [10, 20, 30]
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 0.5
    assert math.isnan(result.iloc[2])


# 時刻差を切り捨てず、1日半を1.5日として扱う。
def test_elapsed_days_support_fractional_days():
    result = calculate_elapsed_days(
        ["2026-01-03 12:00:00"],
        ["2026-01-02 00:00:00"],
    )
    assert result.iloc[0] == 1.5


# 対象日時より後の履歴を与えた場合、未来情報として明示的に拒否する。
def test_future_history_is_rejected():
    try:
        calculate_elapsed_days(["2026-01-01"], ["2026-01-02"])
    except ValueError as exc:
        assert "Future records" in str(exc)
    else:
        raise AssertionError("Future history was accepted")


# ゼロ・負値など不正な半減期を、計算に進む前に拒否する。
def test_invalid_half_life_is_rejected():
    for half_life in (0, -1, float("inf")):
        try:
            decay_weights([1, 2], half_life)
        except ValueError as exc:
            assert "half_life_days" in str(exc)
        else:
            raise AssertionError(f"Invalid half-life was accepted: {half_life}")


# 値または重みが欠損した組を除外して重み付き和・平均を計算し、有効件数も確認する。
def test_weighted_aggregates_ignore_missing_value_pairs():
    values = [80.0, 70.0, None]
    # 経過日数に応じて減衰した観測の重み。加重平均と有効件数の計算に使う。
    weights = [1.0, 0.5, 0.25]

    assert weighted_sum(values, weights) == 115.0
    assert weighted_mean(values, weights) == 115.0 / 1.5
    assert effective_count(weights) == 1.75


# 重み合計がゼロのとき、平均をゼロとせずNaNにする。
def test_weighted_mean_without_effective_weight_is_missing():
    assert math.isnan(weighted_mean([10.0, 20.0], [0.0, 0.0]))
