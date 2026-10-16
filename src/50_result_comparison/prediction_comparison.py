# 保存済み予想と確定着順を照合し、推奨買い目をJRA公式払戻・返還情報で精算する。
# 未確認の結果と取消・除外・中止を区別し、未確定を不的中やゼロ円へ勝手に置き換えない。
# 日別照合はレースごとに失敗を記録して成功分を保持し、入力の予想表を変更せず結果を返す。
#
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from importlib import import_module

import pandas as pd


ResultFetcher = Callable[[str, date], pd.DataFrame]

# 完走・取消・除外・中止などの内部状態と日本語表示の対応。
RESULT_STATUS_LABELS = {
    "finished": "完走", "scratched": "取消", "excluded": "除外",
    "did_not_finish": "中止", "disqualified": "失格", "unknown": "不明",
}


# 予想に載らず買い目にも関係しない取消・除外馬を、結果照合で許容する候補として取り出す。
def withdrawals_without_exposure(actual: pd.DataFrame, prediction: pd.DataFrame, bets: pd.DataFrame) -> set:
    """Withdrawals absent from predictions and not needed by any saved ticket."""
    statuses = actual.get("result_status", pd.Series("unknown", index=actual.index))
    withdrawn = actual.loc[statuses.isin({"scratched", "excluded"})]
    allowed = set(withdrawn.horse_number) - set(prediction.horse_number)
    for bet in bets.itertuples():
        selections = [int(value) for value in str(bet.selection).split("-")]
        if bet.bet_type != "bracket_quinella":
            allowed.difference_update(selections)
        else:
            if "frame_number" not in actual:
                return set()
            active = actual.loc[~statuses.isin({"scratched", "excluded"})]
            for frame in set(selections):
                if active.frame_number.eq(frame).sum() < selections.count(frame):
                    allowed.difference_update(withdrawn.loc[withdrawn.frame_number.eq(frame), "horse_number"])
    return allowed


# 着順と状態の組合せを検査する。未知の状態や着順欠損を完走扱いして通さない。
def validated_result_status(actual: pd.DataFrame, allowed_unlisted: set | None = None) -> pd.Series:
    """Accept known non-finishers, but never silently accept missing results."""
    # allowed_unlisted: 予想にも買い目にも不要と確認できた取消・除外馬の許容集合。
    # 着順を数値化したSeries。取消などは数値順位へ推測しない。
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
    # 取消・除外などの公式情報から判定した返還対象、またはその番号集合。
    refunded = set(refunds.get("horse_numbers", []))
    if not refunded <= withdrawn or not (withdrawn - refunded) <= (allowed_unlisted or set()):
        raise ValueError("取消・除外馬と公式返還馬番が一致しません。手動確認が必要です。")
    return statuses


# 保存済み買い目を公式払戻・返還情報で精算する。元の並びを保ち、未確認は欠損として残す。
def compare_recommended_bets(
    bets: pd.DataFrame,
    official_payouts: dict[str, dict[str, int]],
    official_refunds: dict[str, dict[str, list[int]]] | None = None,
) -> pd.DataFrame:
    """Judge recommended tickets using official payouts per 100 yen."""
    # official_payouts: レース・券種・組合せをキーとする公式の100円当たり払戻金。
    # official_refunds: 馬番・枠番・同枠組合せに関する公式返還情報。
    result = bets.copy()
    # 買い目と同じ順に積む精算ラベル。
    judgments = []
    payouts = []
    for bet in result.itertuples():
        # 現在精算しているレースの公式払戻一覧。別レースの払戻と混在させない。
        race_payouts = official_payouts.get(str(bet.race_id), {})
        bet_type = str(getattr(bet, "bet_type", ""))
        refunds = (official_refunds or {}).get(str(bet.race_id), {})
        selections = [int(value) for value in str(bet.selection).split("-")]
        # 枠連は馬番ではなく枠番で精算する。同枠だけの返還を全枠組合せへ広げない。
        if bet_type == "bracket_quinella":
            # 取消・除外などの公式情報から判定した返還対象、またはその番号集合。
            refunded = bool(set(selections).intersection(refunds.get("frame_numbers", []))) or (
                len(selections) == 2 and selections[0] == selections[1]
                and selections[0] in refunds.get("same_frame_numbers", [])
            )
        else:
            # 枠連以外は馬番による返還情報で判定し、枠番の返還と混在させない。
            refunded = bet_type in {"win", "place", "quinella", "wide", "exacta", "trio", "trifecta"} and bool(
                set(selections).intersection(refunds.get("horse_numbers", []))
            )
        # 公式返還対象の買い目は100円を返し、的中・不的中とは別の判定で次の買い目へ進む。
        if refunded:
            judgments.append("返還")
            payouts.append(100)
            continue
        # 券種の払戻情報がない場合は不的中と断定せず、未確認と払戻欠損で残す。
        if not bet_type or not any(key.startswith(f"{bet_type}:") for key in race_payouts):
            judgments.append("未確認")
            payouts.append(pd.NA)
            continue
        # 順序を問わない券種は選択番号を並べ替え、公式払戻のキーと同じ形式へそろえる。
        if bet_type in {"bracket_quinella", "quinella", "wide", "trio"}:
            selections.sort()
        key = f'{bet_type}:{"-".join(map(str, selections))}'
        judgments.append("的中" if key in race_payouts else "不的中")
        payouts.append(race_payouts.get(key, 0))
    result["bet_result"] = judgments
    result["payout_per_100"] = pd.array(payouts, dtype="Int64")
    plan = import_module('src.00_common.purchase_plan')
    if set(plan.PLAN_COLUMNS) & set(result):
        result = plan.validate_plan(result)
    elif 'stake_yen' not in result:
        result['stake_yen'] = plan.STAKE
    stakes = pd.to_numeric(result.stake_yen, errors='coerce')
    if not (stakes.gt(0) & stakes.mod(100).eq(0)).all():
        raise ValueError('Invalid settlement stake')
    result['payout_yen'] = result.payout_per_100 * stakes / 100
    result['profit_yen'] = result.payout_yen - stakes
    return result


# 予想と確定結果を馬単位で対応付け、非完走馬の状態を保ったまま上位3頭の的中を計算する。
def compare_prediction_with_finish(
    prediction: pd.DataFrame,
    actual: pd.DataFrame,
    allowed_unlisted: set | None = None,
) -> pd.DataFrame:
    """Join one race prediction with its confirmed finish positions."""

    # allowed_unlisted: 予想にも買い目にも不要と確認できた取消・除外馬の許容集合。
    required_prediction = {"horse_id", "horse_number", "prediction_rank"}
    missing = required_prediction.difference(prediction.columns)
    # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
    if missing:
        raise ValueError(f"予想結果に比較用の列がありません: {sorted(missing)}")
    required_actual = {"horse_id", "horse_number", "finish_position"}
    missing = required_actual.difference(actual.columns)
    # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
    if missing:
        raise ValueError(f"結果HTMLに着順列がありません: {sorted(missing)}")

    finish = actual.loc[:, list(required_actual)].copy()
    finish["result_status"] = validated_result_status(actual, allowed_unlisted)
    finish["finish_position"] = pd.to_numeric(
        finish["finish_position"], errors="coerce"
    )
    # 予想と公式結果の両方に馬IDがあり、そのIDで照合できるかの判定。
    use_horse_id = (
        prediction["horse_id"].notna().any()
        and finish["horse_id"].notna().any()
    )
    # 馬IDが利用可能なら馬ID、利用できなければ馬番を採用する照合キー。
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


# レースごとに結果取得と照合を行い、失敗内容を集約しながら成功分の表を保持する。
def compare_prediction_date(
    predictions: pd.DataFrame,
    race_date: date,
    fetch_result: ResultFetcher,
    bets: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fetch and compare every predicted race on a date, retaining partial success."""

    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    # fetch_result: レースIDと開催日から確定結果を返す取得関数。通信を差し替えられる。
    if predictions.empty:
        raise ValueError("比較する予想結果がありません。")
    if "race_id" not in predictions:
        raise ValueError("予想結果にrace_idがありません。")

    # 取得・検証に成功したレースだけの比較表。最後に連結する。
    comparisons: list[pd.DataFrame] = []
    # 失敗したレースIDと原因を残す一覧。成功分を失わず不足を報告する。
    failures: list[dict[str, str]] = []
    official_payouts: dict[str, dict[str, int]] = {}
    official_refunds: dict[str, dict[str, list[int]]] = {}
    # レース単位で処理し、別レースの馬番や払戻金を混ぜない。
    for race_id, race_prediction in predictions.groupby("race_id", sort=False):
        try:
            actual = fetch_result(str(race_id), race_date)
            allowed = withdrawals_without_exposure(actual, race_prediction, bets[bets.race_id.astype(str).eq(str(race_id))]) if bets is not None else set()
            comparisons.append(
                compare_prediction_with_finish(race_prediction.copy(), actual, allowed)
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
        # レースごとの予測上位3頭の的中頭数。
        hits_by_race = comparison.groupby("race_id")["top3_hit"].sum()
        # 照合済みレースで、予測上位3頭に入れた馬が実際にも3着内だった延べ頭数。
        hit_count = int(comparison["top3_hit"].sum())
        # 予測上位3頭がすべて的中したレース数。
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
