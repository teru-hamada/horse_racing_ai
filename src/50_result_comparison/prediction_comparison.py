from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd


ResultFetcher = Callable[[str, date], pd.DataFrame]

RESULT_STATUS_LABELS = {
    "finished": "完走", "scratched": "取消", "excluded": "除外",
    "did_not_finish": "中止", "disqualified": "失格", "unknown": "不明",
}


def validated_result_status(actual: pd.DataFrame) -> pd.Series:
    """Accept known non-finishers, but never silently accept missing results."""
    positions = pd.to_numeric(actual.finish_position, errors="coerce")
    statuses = actual.get("result_status")
    if statuses is None:
        statuses = pd.Series("unknown", index=actual.index)
        statuses.loc[positions.gt(0)] = "finished"
    valid = (statuses.eq("finished") & positions.gt(0) & positions.mod(1).eq(0)) | (
        statuses.isin({"scratched", "excluded", "did_not_finish", "disqualified"})
        & positions.isna()
    )
    if not valid.fillna(False).all() or not (statuses.eq("finished") & positions.eq(1)).any():
        raise ValueError("確定着順または出走結果の状態が不明です。手動確認が必要です。")
    withdrawn = set(actual.loc[statuses.isin({"scratched", "excluded"}), "horse_number"])
    refunds = actual.attrs.get("official_refunds", {})
    if withdrawn != set(refunds.get("horse_numbers", [])):
        raise ValueError("取消・除外馬と公式返還馬番が一致しません。手動確認が必要です。")
    return statuses


def compare_recommended_bets(
    bets: pd.DataFrame,
    official_payouts: dict[str, dict[str, int]],
    official_refunds: dict[str, dict[str, list[int]]] | None = None,
) -> pd.DataFrame:
    """Judge recommended tickets using official payouts per 100 yen."""
    result = bets.copy()
    judgments = []
    payouts = []
    for bet in result.itertuples():
        race_payouts = official_payouts.get(str(bet.race_id), {})
        bet_type = str(getattr(bet, "bet_type", ""))
        refunds = (official_refunds or {}).get(str(bet.race_id), {})
        selections = [int(value) for value in str(bet.selection).split("-")]
        if bet_type == "bracket_quinella":
            refunded = bool(set(selections).intersection(refunds.get("frame_numbers", []))) or (
                len(selections) == 2 and selections[0] == selections[1]
                and selections[0] in refunds.get("same_frame_numbers", [])
            )
        else:
            refunded = bet_type in {"win", "place", "quinella", "wide", "exacta", "trio", "trifecta"} and bool(
                set(selections).intersection(refunds.get("horse_numbers", []))
            )
        if refunded:
            judgments.append("返還")
            payouts.append(100)
            continue
        if not bet_type or not any(key.startswith(f"{bet_type}:") for key in race_payouts):
            judgments.append("未確認")
            payouts.append(pd.NA)
            continue
        if bet_type in {"bracket_quinella", "quinella", "wide", "trio"}:
            selections.sort()
        key = f'{bet_type}:{"-".join(map(str, selections))}'
        judgments.append("的中" if key in race_payouts else "不的中")
        payouts.append(race_payouts.get(key, 0))
    result["bet_result"] = judgments
    result["payout_per_100"] = pd.array(payouts, dtype="Int64")
    return result


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
    finish["result_status"] = validated_result_status(actual)
    finish["finish_position"] = pd.to_numeric(
        finish["finish_position"], errors="coerce"
    )
    use_horse_id = (
        prediction["horse_id"].notna().any()
        and finish["horse_id"].notna().any()
    )
    join_key = "horse_id" if use_horse_id else "horse_number"
    finish = finish.dropna(subset=[join_key]).drop_duplicates(join_key, keep="last")
    comparison = prediction.merge(
        finish[[join_key, "finish_position", "result_status"]],
        on=join_key,
        how="left",
        validate="many_to_one",
    )
    comparison["predicted_top3"] = comparison["prediction_rank"].le(3)
    comparison["actual_top3"] = (
        comparison["result_status"].eq("finished") & comparison["finish_position"].le(3)
    ).fillna(False)
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
    official_payouts: dict[str, dict[str, int]] = {}
    official_refunds: dict[str, dict[str, list[int]]] = {}
    for race_id, race_prediction in predictions.groupby("race_id", sort=False):
        try:
            actual = fetch_result(str(race_id), race_date)
            comparisons.append(
                compare_prediction_with_finish(race_prediction.copy(), actual)
            )
            payouts = actual.attrs.get("official_payouts", {})
            if payouts:
                official_payouts[str(race_id)] = dict(payouts)
            refunds = actual.attrs.get("official_refunds", {})
            if refunds:
                official_refunds[str(race_id)] = dict(refunds)
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
        "official_payouts": official_payouts,
        "official_refunds": official_refunds,
    }
    return comparison, summary
