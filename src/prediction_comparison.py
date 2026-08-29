from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd


ResultFetcher = Callable[[str, date], pd.DataFrame]


def compare_prediction_with_finish(
    prediction: pd.DataFrame,
    actual: pd.DataFrame,
) -> pd.DataFrame:
    """Join one race prediction with its confirmed finish positions."""

    required_prediction = {"horse_id", "horse_number", "prediction_rank"}
    missing = required_prediction.difference(prediction.columns)
    if missing:
        raise ValueError(f"予想結果に比較用の列がありません: {sorted(missing)}")
    required_actual = {"horse_id", "horse_number", "finish_position"}
    missing = required_actual.difference(actual.columns)
    if missing:
        raise ValueError(f"結果HTMLに着順列がありません: {sorted(missing)}")

    finish = actual.loc[:, list(required_actual)].copy()
    finish["finish_position"] = pd.to_numeric(
        finish["finish_position"], errors="coerce"
    )
    finish = finish.dropna(subset=["finish_position"])
    if finish.empty:
        raise ValueError("確定着順がありません。結果確定後に再実行してください。")

    use_horse_id = (
        prediction["horse_id"].notna().any()
        and finish["horse_id"].notna().any()
    )
    join_key = "horse_id" if use_horse_id else "horse_number"
    finish = finish.dropna(subset=[join_key]).drop_duplicates(join_key, keep="last")
    comparison = prediction.merge(
        finish[[join_key, "finish_position"]],
        on=join_key,
        how="left",
        validate="many_to_one",
    )
    comparison["predicted_top3"] = comparison["prediction_rank"].le(3)
    comparison["actual_top3"] = comparison["finish_position"].le(3)
    comparison["top3_hit"] = (
        comparison["predicted_top3"] & comparison["actual_top3"]
    )
    return comparison


def compare_prediction_date(
    predictions: pd.DataFrame,
    race_date: date,
    fetch_result: ResultFetcher,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fetch and compare every predicted race on a date, retaining partial success."""

    if predictions.empty:
        raise ValueError("比較する予想結果がありません。")
    if "race_id" not in predictions:
        raise ValueError("予想結果にrace_idがありません。")

    comparisons: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for race_id, race_prediction in predictions.groupby("race_id", sort=False):
        try:
            actual = fetch_result(str(race_id), race_date)
            comparisons.append(
                compare_prediction_with_finish(race_prediction.copy(), actual)
            )
        except Exception as exc:  # one unavailable result must not hide other races
            failures.append({"race_id": str(race_id), "message": str(exc)})

    comparison = (
        pd.concat(comparisons, ignore_index=True)
        if comparisons
        else pd.DataFrame()
    )
    if not comparison.empty:
        comparison = comparison.sort_values(
            ["course_name", "race_number", "race_id", "prediction_rank"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)
        hits_by_race = comparison.groupby("race_id")["top3_hit"].sum()
        hit_count = int(comparison["top3_hit"].sum())
        perfect_races = int(hits_by_race.eq(3).sum())
    else:
        hit_count = 0
        perfect_races = 0
    summary: dict[str, object] = {
        "requested_races": int(predictions["race_id"].nunique()),
        "compared_races": len(comparisons),
        "failed_races": len(failures),
        "top3_hit_count": hit_count,
        "perfect_top3_races": perfect_races,
        "failures": failures,
    }
    return comparison, summary
