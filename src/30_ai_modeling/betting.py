from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd


BET_TYPE_LABELS = {
    "win": "単勝", "place": "複勝", "bracket_quinella": "枠連", "quinella": "馬連",
    "wide": "ワイド", "exacta": "馬単", "trio": "3連複",
    "trifecta": "3連単",
}

# 現行モデルは「3着以内確率」を予測するため、的中条件が直接一致する
# 複勝を1.0とし、着順・同時確率の近似が増える券種ほど割り引く。
BET_TYPE_RELIABILITY = {
    "place": 1.00,
    "wide": 0.85,
    "win": 0.75,
    "quinella": 0.70,
    "bracket_quinella": 0.70,
    "exacta": 0.60,
    "trio": 0.55,
    "trifecta": 0.40,
}


def _ordered_probability(order: tuple[int, ...], weights: dict[int, float]) -> float:
    remaining = 1.0
    probability = 1.0
    for horse in order:
        weight = weights.get(horse, 0.0)
        if remaining <= 0 or weight <= 0:
            return 0.0
        probability *= weight / remaining
        remaining -= weight
    return float(probability)


def _estimated_probability(
    bet_type: str,
    selections: tuple[int, ...],
    weights: dict[int, float],
    top3: dict[int, float],
) -> tuple[float, str]:
    if bet_type == "place":
        return float(np.clip(top3.get(selections[0], 0.0), 0.0, 1.0)), "3着内確率（モデル出力）"
    if bet_type == "win":
        return weights.get(selections[0], 0.0), "Plackett-Luce近似"
    if bet_type == "exacta":
        return _ordered_probability(selections, weights), "Plackett-Luce近似"
    if bet_type == "quinella":
        return sum(
            _ordered_probability(order, weights)
            for order in permutations(selections)
        ), "Plackett-Luce近似"
    if bet_type == "trifecta":
        return _ordered_probability(selections, weights), "Plackett-Luce近似"
    if bet_type == "trio":
        return sum(
            _ordered_probability(order, weights)
            for order in permutations(selections)
        ), "Plackett-Luce近似"
    if bet_type == "wide":
        others = set(weights).difference(selections)
        probability = 0.0
        for third in others:
            probability += sum(
                _ordered_probability(order, weights)
                for order in permutations((*selections, third))
            )
        return probability, "Plackett-Luce近似"
    return 0.0, "対象外"


def calculate_bet_recommendations(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    best_only: bool = True,
) -> pd.DataFrame:
    """予測後にだけオッズを結合し、100円当たりの期待値を計算する。"""
    columns = [
        "race_id", "course_name", "race_number", "race_name", "bet_type",
        "bet_type_label", "selection", "estimated_probability", "odds_used",
        "recovery_rate_percent", "bet_type_reliability", "recommendation_score",
        "expected_profit_per_100", "probability_method",
    ]
    if predictions.empty or odds.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    supported = set(BET_TYPE_LABELS)
    for race_id, race_predictions in predictions.groupby("race_id", sort=False):
        race_odds = odds[
            odds["race_id"].astype(str).eq(str(race_id))
            & odds["bet_type"].isin(supported)
        ]
        if race_odds.empty:
            continue
        probability_by_horse = {
            int(row.horse_number): float(row.top3_probability)
            for row in race_predictions.itertuples()
            if pd.notna(row.horse_number) and pd.notna(row.top3_probability)
        }
        total = sum(max(value, 0.0) for value in probability_by_horse.values())
        if total <= 0:
            continue
        weights = {
            horse: max(value, 0.0) / total
            for horse, value in probability_by_horse.items()
        }
        frame_by_horse = {
            int(row.horse_number): int(row.frame_number)
            for row in race_predictions.itertuples()
            if hasattr(row, "frame_number")
            and pd.notna(row.horse_number)
            and pd.notna(row.frame_number)
        }
        first = race_predictions.iloc[0]
        for odd in race_odds.itertuples():
            selections = tuple(
                int(value)
                for value in (odd.selection_1, odd.selection_2, odd.selection_3)
                if pd.notna(value)
            )
            if not selections or any(value not in weights for value in selections):
                if str(odd.bet_type) != "bracket_quinella":
                    continue
            if str(odd.bet_type) == "bracket_quinella":
                if len(selections) != 2 or not frame_by_horse:
                    continue
                target_frames = tuple(sorted(selections))
                probability = sum(
                    _ordered_probability(order, weights)
                    for order in permutations(weights, 2)
                    if tuple(sorted(frame_by_horse.get(horse, -1) for horse in order))
                    == target_frames
                )
                method = "Plackett-Luce近似（枠集計）"
            else:
                probability, method = _estimated_probability(
                    str(odd.bet_type), selections, weights, probability_by_horse
                )
            odds_used = float(odd.odds_min) if pd.notna(odd.odds_min) else np.nan
            if not np.isfinite(odds_used) or odds_used <= 0:
                continue
            recovery_rate_percent = probability * odds_used * 100.0
            reliability = BET_TYPE_RELIABILITY[str(odd.bet_type)]
            rows.append(
                {
                    "race_id": str(race_id),
                    "course_name": first.get("course_name"),
                    "race_number": first.get("race_number"),
                    "race_name": first.get("race_name"),
                    "bet_type": str(odd.bet_type),
                    "bet_type_label": BET_TYPE_LABELS[str(odd.bet_type)],
                    "selection": "-".join(map(str, selections)),
                    "estimated_probability": probability,
                    "odds_used": odds_used,
                    "recovery_rate_percent": recovery_rate_percent,
                    "bet_type_reliability": reliability,
                    "recommendation_score": recovery_rate_percent * reliability,
                    "expected_profit_per_100": recovery_rate_percent - 100.0,
                    "probability_method": method,
                }
            )
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    result = result.sort_values(
        ["race_id", "recommendation_score", "recovery_rate_percent"],
        ascending=[True, False, False],
        kind="stable",
    )
    if best_only:
        result = result.groupby("race_id", sort=False).head(1)
    return result.reset_index(drop=True)
