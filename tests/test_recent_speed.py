# 検証対象：過去の有効な速度指数だけで直近統計と時間減衰統計を作ることを確認する。
# 対象走自身の指数が混ざらず、履歴なしの馬は欠損値と件数ゼロになることを検証する。
#
import math

import pandas as pd

from src.public_api import FeatureContext, RecentSpeedGenerator


# 検証に必要な入力表の組を作り、保存前後の照合へ渡す。
def _frames():
    targets = pd.DataFrame([
        {"race_id": f"r{i}", "horse_id": "h1", "race_date": f"2026-01-{i:02d}"}
        for i in range(1, 7)
    ])
    performance = targets.copy()
    performance["speed_index"] = [10, 20, 30, 40, 50, 60]
    return targets, performance


# 過去の指数から直近平均・最大・標準偏差・件数を求め、同日以降の値を含めない。
def test_recent_speed_aggregates_only_prior_indexes():
    targets, performance = _frames()
    result = RecentSpeedGenerator().transform(
        targets, performance, FeatureContext("recent_speed", "1.0.0", {})
    )
    row = result[result["race_id"].eq("r6")].iloc[0]

    assert row["recent_speed_last"] == 50
    assert row["recent_speed_mean_3"] == 40
    assert row["recent_speed_mean_5"] == 30
    assert row["recent_speed_max_5"] == 50
    assert row["recent_speed_std_5"] == math.sqrt(200)
    assert row["recent_speed_available_count_5"] == 5
    assert 4.9 < row["recent_speed_effective_count"] < 5.0


# 対象レース自身の速度指数を変えても、事前の近走速度特徴量は変わらない。
def test_current_race_speed_index_does_not_affect_its_features():
    targets, performance = _frames()
    context = FeatureContext("recent_speed", "1.0.0", {})
    before = RecentSpeedGenerator().transform(targets, performance, context)
    performance.loc[performance["race_id"].eq("r6"), "speed_index"] = 200
    after = RecentSpeedGenerator().transform(targets, performance, context)

    pd.testing.assert_series_equal(
        before[before["race_id"].eq("r6")].iloc[0],
        after[after["race_id"].eq("r6")].iloc[0],
    )


# 指数履歴がない馬は直近値がNaNで、有効件数がゼロになる。
def test_horse_without_speed_history_has_zero_counts():
    targets, performance = _frames()
    new_target = pd.DataFrame([{
        "race_id": "new", "horse_id": "new-horse", "race_date": "2026-01-07"
    }])
    row = RecentSpeedGenerator().transform(
        new_target, performance, FeatureContext("recent_speed", "1.0.0", {})
    ).iloc[0]

    assert math.isnan(row["recent_speed_last"])
    assert row["recent_speed_available_count_5"] == 0
    assert row["recent_speed_effective_count"] == 0
