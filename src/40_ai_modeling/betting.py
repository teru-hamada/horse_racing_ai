# モデルの3着内確率とJRAオッズを予想後に結合し、券種ごとの買い目候補を評価する。
# 複勝以外の確率には着順の近似を含むため、券種別の信頼度と最低確率で候補を絞る。
# 期待回収率・100円当たり期待損益は予測確率からの計算値で、確定払戻の精算とは別に扱う。
#
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd


# 内部の券種識別子から画面表示用の日本語名への対応。
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

# 極端に低確率の候補を除く、券種ごとの推定的中確率の下限。
MIN_HIT_PROBABILITY = {
    "place": 0.25,
    "win": 0.10,
    "wide": 0.08,
    "quinella": 0.05,
    "bracket_quinella": 0.05,
    "exacta": 0.03,
    "trio": 0.02,
    "trifecta": 0.01,
}
# 候補に残す期待回収率の下限（%）。実際の回収を保証する値ではない。
MIN_RECOVERY_RATE_PERCENT = 110.0
# 推奨順位で優先する券種。モデル出力との対応を重視する。
PRIMARY_BET_TYPES = {"place", "wide"}


# 選ばれた馬の重みを残りの重みで順に割り、指定着順の同時確率を近似する。
def _ordered_probability(order: tuple[int, ...], weights: dict[int, float]) -> float:
    # まだ着順に選ばれていない馬の重み合計。選択のたびに減らす。
    remaining = 1.0
    # 指定順序の確率を積み上げる値。各段階の条件付き確率を掛け合わせる。
    probability = 1.0
    for horse in order:
        weight = weights.get(horse, 0.0)
        if remaining <= 0 or weight <= 0:
            return 0.0
        probability *= weight / remaining
        remaining -= weight
    return float(probability)


# 券種の順序条件に応じて順列を合算する。複勝はモデルの3着内確率を直接使う。
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


# 予測確率とオッズから期待値を算出し、最低確率・回収率・券種別信頼度で推奨候補を選ぶ。
def calculate_bet_recommendations(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    best_only: bool = True,
    include_rejected: bool = False,
) -> pd.DataFrame:
    """予測後にだけオッズを結合し、100円当たりの期待値を計算する。"""
    # best_only: 推奨候補を絞り込むか、条件に合う候補一覧を返すかの指定。
    # include_rejected: 監査用。閾値適用前の全行を返し、上位3点の採否と理由を付与する。
    columns = [
        "race_id", "course_name", "race_number", "race_name", "bet_type",
        "bet_type_label", "selection", "estimated_probability", "odds_used",
        "recovery_rate_percent", "bet_type_reliability", "recommendation_score",
        "expected_profit_per_100", "probability_method", "is_primary_bet_type",
    ]
    # 予想またはオッズがなければ期待値を計算せず、既定列を持つ空の候補表を返す。
    if (odds.empty or predictions.empty) and not include_rejected:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    supported = set(BET_TYPE_LABELS)
    # レース単位で処理し、別レースの馬番や払戻金を混ぜない。
    for race_id, race_odds in odds.groupby("race_id", sort=False):
        race_predictions = predictions[predictions["race_id"].astype(str).eq(str(race_id))]
        # オッズ側を基準にするため、予測のないレースも監査記録へ残せる。
        if race_predictions.empty:
            if include_rejected:
                for odd in race_odds.itertuples():
                    rows.append(_rejected_row(odd, "missing_prediction"))
            continue
        probability_by_horse = {
            int(row.horse_number): float(row.top3_probability)
            for row in race_predictions.itertuples()
            if pd.notna(row.horse_number) and pd.notna(row.top3_probability)
        }
        total = sum(max(value, 0.0) for value in probability_by_horse.values())
        if total <= 0:
            if include_rejected:
                rows.extend(_rejected_row(odd, "invalid_probability") for odd in race_odds.itertuples())
            continue
        # モデルの3着内確率を正規化した着順近似用の重み。市場オッズから作る値ではない。
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
            if include_rejected and not getattr(odd, 'prediction_input', True):
                rows.append(_rejected_row(odd, "excluded_from_prediction_inputs"))
                continue
            if str(odd.bet_type) not in supported:
                if include_rejected:
                    rows.append(_rejected_row(odd, "unsupported_bet_type"))
                continue
            selections = tuple(
                int(value)
                for value in (odd.selection_1, odd.selection_2, odd.selection_3)
                if pd.notna(value)
            )
            expected_size = {"win": 1, "place": 1, "trio": 3, "trifecta": 3}.get(str(odd.bet_type), 2)
            if len(selections) != expected_size or (str(odd.bet_type) != "bracket_quinella" and len(set(selections)) != len(selections)):
                if include_rejected:
                    rows.append(_rejected_row(odd, "invalid_selection"))
                continue
            if not selections or any(value not in weights for value in selections):
                if str(odd.bet_type) != "bracket_quinella":
                    if include_rejected:
                        rows.append(_rejected_row(odd, "missing_prediction"))
                    continue
            if str(odd.bet_type) == "bracket_quinella":
                if len(selections) != 2 or not frame_by_horse:
                    if include_rejected:
                        rows.append(_rejected_row(odd, "missing_frame_prediction"))
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
            # 期待値計算に採用したオッズ。上下限のある券種では選択規則を適用した値。
            odds_used = float(odd.odds_min) if pd.notna(odd.odds_min) else np.nan
            if not np.isfinite(odds_used) or odds_used <= 0:
                if include_rejected:
                    rows.append(_rejected_row(odd, "invalid_odds"))
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
                    "is_primary_bet_type": str(odd.bet_type) in PRIMARY_BET_TYPES,
                }
            )
    if include_rejected:
        result = pd.DataFrame(rows).reindex(columns=columns + ["rejection_reason"])
        result["minimum_hit_probability"] = result.bet_type.map(MIN_HIT_PROBABILITY)
        result["minimum_recovery_rate_percent"] = MIN_RECOVERY_RATE_PERCENT
        result["rejection_reason"] = result.rejection_reason.fillna("")
        for index, row in result.iterrows():
            if row.rejection_reason:
                continue
            reasons = []
            if row.estimated_probability < row.minimum_hit_probability:
                reasons.append("low_probability")
            if row.recovery_rate_percent < MIN_RECOVERY_RATE_PERCENT:
                reasons.append("low_expected_return")
            result.at[index, "rejection_reason"] = "|".join(reasons)
        result["eligible"] = result.rejection_reason.eq("")
        result["recommendation_rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        eligible = result[result.eligible].sort_values(
            ["race_id", "is_primary_bet_type", "recommendation_score", "recovery_rate_percent"],
            ascending=[True, False, False, False], kind="stable").sort_values(
                ["recommendation_score", "recovery_rate_percent"], ascending=False, kind="stable")
        from importlib import import_module
        plan = import_module('src.00_common.purchase_plan').select_plan(eligible)
        result.loc[eligible.index, "recommendation_rank"] = eligible.groupby("race_id", sort=False).cumcount() + 1
        chosen_keys = set(zip(plan.race_id.astype(str), plan.bet_type, plan.selection))
        result["selected"] = [bool(row.eligible and (str(row.race_id), row.bet_type, row.selection) in chosen_keys)
                              for row in result.itertuples()]
        result.loc[result.eligible & ~result.selected, "rejection_reason"] = "outside_top3"
        result["selection_reason"] = np.where(result.selected, "thresholds_passed_and_top3", "")
        return result.reset_index(drop=True)
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    result = result[
        result.apply(
            lambda row: (
                row["estimated_probability"]
                >= MIN_HIT_PROBABILITY.get(str(row["bet_type"]), 1.0)
                and row["recovery_rate_percent"] >= MIN_RECOVERY_RATE_PERCENT
            ),
            axis=1,
        )
    ].copy()
    if result.empty:
        return result.reset_index(drop=True)
    result = result.sort_values(
        [
            "race_id", "is_primary_bet_type", "recommendation_score",
            "recovery_rate_percent",
        ],
        ascending=[True, False, False, False],
        kind="stable",
    )
    if best_only:
        result = result.groupby("race_id", sort=False).head(1)
    return result.reset_index(drop=True)


def _rejected_row(odd, reason):
    """計算不能な買い目も理由付きで残す。未計算の確率・期待値は欠損値とする。"""
    selections = [int(v) for v in (odd.selection_1, odd.selection_2, odd.selection_3) if pd.notna(v)]
    return {"race_id": str(odd.race_id), "bet_type": str(odd.bet_type),
            "selection": "-".join(map(str, selections)), "rejection_reason": reason,
            "odds_used": odd.odds_min}
