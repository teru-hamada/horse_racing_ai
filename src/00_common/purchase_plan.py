"""購入対象を予想時に固定し、表示・精算で共有する。過去の候補CSVも同じ規則で読む。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PLAN_COLUMNS = ('purchase_plan_version', 'purchase_rank', 'stake_yen')
LIMIT = 3
STAKE = 100


def validate_plan(bets):
    """保存済み購入対象は並べ替え・再選定せず検証する。金額は100円単位。"""
    if not set(PLAN_COLUMNS) <= set(bets):
        raise ValueError('Incomplete purchase plan columns')
    if bets.empty:
        return bets.copy()
    if bets[['race_id', 'bet_type', 'selection']].isna().any().any():
        raise ValueError('Missing purchase ticket identity')
    if bets.duplicated(['race_id', 'bet_type', 'selection']).any():
        raise ValueError('Duplicate purchase tickets')
    for column in PLAN_COLUMNS:
        values = pd.to_numeric(bets[column], errors='coerce')
        if not np.isfinite(values).all() or not values.mod(1).eq(0).all():
            raise ValueError('Invalid purchase plan numbers')
    if not pd.to_numeric(bets.purchase_plan_version).eq(1).all():
        raise ValueError('Unsupported purchase plan version')
    stakes = pd.to_numeric(bets.stake_yen)
    if not (stakes.gt(0) & stakes.mod(100).eq(0)).all():
        raise ValueError('Stake must be positive multiples of 100 yen')
    for _, race in bets.groupby('race_id', sort=False):
        if len(race) > LIMIT or sorted(pd.to_numeric(race.purchase_rank)) != list(range(1, len(race) + 1)):
            raise ValueError('Invalid purchase ranks or ticket count')
    result = bets.copy()
    for column in PLAN_COLUMNS:
        result[column] = pd.to_numeric(result[column]).astype(int)
    return result


def adopt_saved_plan(bets):
    """旧形式で既に保存された上位3点に、金額と版番号だけを付ける。再選定しない。"""
    if set(PLAN_COLUMNS) & set(bets):
        return validate_plan(bets)
    result = bets.copy()
    result['purchase_plan_version'] = 1
    result['purchase_rank'] = result.groupby('race_id', sort=False).cumcount() + 1
    result['stake_yen'] = STAKE
    return validate_plan(result)


def select_plan(candidates):
    """共通の購入対象選定。スコア・期待回収率の順に各レース上位3点を固定する。"""
    if set(PLAN_COLUMNS) & set(candidates):
        return validate_plan(candidates)
    if candidates.empty:
        return adopt_saved_plan(candidates)
    chosen = candidates.sort_values(['recommendation_score', 'recovery_rate_percent'],
                                    ascending=False, kind='stable').groupby('race_id', sort=False).head(LIMIT)
    return adopt_saved_plan(chosen.reset_index(drop=True))


def resolve_plan(bets):
    """新形式はそのまま使用する。候補一覧だった旧形式だけ共通規則で変換する。"""
    return select_plan(bets)


def load_saved_plan(prediction_path, method='none'):
    """保存済み予想に付属する購入対象を読む。現在のDBオッズで作り直さない。"""
    path = Path(prediction_path)
    root = path if path.is_dir() else path.parent
    for source in (root / 'recommended_bets.csv', root / 'calibration' / method / 'recommended_bets.csv'):
        if source.exists():
            return adopt_saved_plan(pd.read_csv(source, dtype={'race_id': str, 'selection': str}))
    if (root / 'bets.csv').exists():
        return resolve_plan(pd.read_csv(root / 'bets.csv', dtype={'race_id': str, 'selection': str}))
    raise FileNotFoundError('Saved purchase plan is unavailable')


def return_summary(judged):
    """保存した金額と実払戻額から回収率を計算する。未確定は回収率を出さない。"""
    known = judged.bet_result.ne('未確認')
    total, settled = len(judged), int(known.sum())
    purchase = int(judged.stake_yen.sum())
    payout = float(judged.loc[known, 'payout_yen'].sum())
    return {'total_bets': total, 'settled_bets': settled,
            'hit_bets': int(judged.bet_result.eq('的中').sum()),
            'refund_bets': int(judged.bet_result.eq('返還').sum()),
            'purchase': purchase, 'payout': payout,
            'recovery_rate': payout / purchase * 100 if purchase and total == settled else None}
