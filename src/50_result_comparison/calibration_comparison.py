"""保存済みの方式別上位買い目を、同一の公式結果で精算する。再選定はしない。"""
from __future__ import annotations

import hashlib
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prediction_comparison import (
    compare_recommended_bets, validated_result_status, withdrawals_without_exposure,
)

METHODS = ('class_weight', 'sigmoid')
KEYS = ['race_id', 'bet_type', 'selection']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None


def snapshot_inputs(predictions_root: Path, day: str, output: Path):
    """取得前の比較入力をコピーする。欠落も記録し、後からの追加・変更も検知する。"""
    hashes = {}
    paths = [Path('calibration') / day / 'status.json']
    paths += [Path('calibration') / day / method / name for method in METHODS
              for name in ('predictions.csv', 'bets.csv', 'recommended_bets.csv')]
    for relative in paths:
        source = predictions_root / relative
        hashes[relative.as_posix()] = digest(source)
        if hashes[relative.as_posix()] is not None:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    return hashes


def validate_snapshot(predictions_root: Path, day: str, hashes):
    """公開直前に全入力を再確認。任意パスを読まないよう固定のパス集合へ限定する。"""
    expected = {f'calibration/{day}/status.json'} | {
        f'calibration/{day}/{method}/{name}' for method in METHODS
        for name in ('predictions.csv', 'bets.csv', 'recommended_bets.csv')}
    if set(hashes) != expected:
        raise ValueError('比較用買い目のハッシュ情報が不正です。')
    for relative, original in hashes.items():
        if digest(predictions_root / relative) != original:
            raise ValueError('補正比較の予想・買い目が照合開始後に変更されました。再実行してください。')


def settle_saved_comparisons(snapshot: Path, day: str, public_predictions, actual_results, output: Path):
    """方式ごとに独立精算し、不足を主方式の失敗や0%へ置き換えない。"""
    root = snapshot / 'calibration' / day
    status_path = root / 'status.json'
    try:
        source_status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        if not isinstance(source_status, dict):
            raise ValueError('Invalid comparison status')
    except (OSError, ValueError):
        source_status = {method: {'status': 'invalid'} for method in METHODS}
    results = {}
    for method in METHODS:
        directory = root / method
        result = {'status': 'missing', 'reason': '比較用の保存済み予想・買い目がありません。'}
        try:
            method_status = source_status.get(method, {})
            if not isinstance(method_status, dict):
                raise ValueError('比較予想の状態情報が不正です。')
            if method_status.get('status') not in (None, 'ok'):
                results[method] = {'status': 'unavailable', 'reason': '比較予想が未作成または一部レース不足です。'}
                continue
            if not (directory / 'predictions.csv').is_file() or not (directory / 'recommended_bets.csv').is_file():
                results[method] = result
                continue
            predictions = pd.read_csv(directory / 'predictions.csv', dtype={'race_id': str, 'horse_id': str})
            bets = pd.read_csv(directory / 'recommended_bets.csv', dtype={'race_id': str, 'selection': str})
            bets = import_module('src.00_common.purchase_plan').adopt_saved_plan(bets)
            identity = ['race_id', 'horse_id', 'horse_number']
            if (predictions.empty or predictions.duplicated(identity).any()
                    or set(map(tuple, predictions[identity].astype(str).to_numpy())) !=
                    set(map(tuple, public_predictions[identity].astype(str).to_numpy()))
                    or not pd.to_datetime(predictions.race_date).dt.date.astype(str).eq(day).all()
                    or set(predictions.model_run_id) != set(public_predictions.model_run_id)
                    or not predictions.calibration_method.eq(method).all()):
                raise ValueError('比較予想の対象馬・開催日・モデル・方式が一致しません。')
            probabilities = predictions.top3_probability.to_numpy(dtype=float)
            if (not np.isfinite(probabilities).all() or not predictions.top3_probability.between(0, 1).all()
                    or predictions.calibration_id.isna().any() or predictions.calibration_id.nunique() != 1):
                raise ValueError('比較予想の確率・補正定義が不正です。')
            if (not set(bets.race_id) <= set(predictions.race_id) or bets.duplicated(KEYS).any()
                    or bets.groupby('race_id').size().gt(3).any()):
                raise ValueError('保存済み比較買い目のレース・点数・重複が不正です。')
            if not bets.empty:
                if not bets.calibration_method.eq(method).all():
                    raise ValueError('比較買い目の方式が一致しません。')
                if not bets.calibration_id.eq(predictions.calibration_id.iloc[0]).all():
                    raise ValueError('比較買い目の補正定義が一致しません。')
                numeric = bets[['estimated_probability', 'odds_used']].to_numpy(dtype=float)
                if (not np.isfinite(numeric).all() or not bets.estimated_probability.between(0, 1).all()
                        or not bets.odds_used.gt(0).all()):
                    raise ValueError('比較買い目の数値が不正です。')
            payouts, refunds, failures = {}, {}, []
            for race_id in bets.race_id.unique():
                if race_id not in actual_results:
                    failures.append(str(race_id))
                    continue
                actual = actual_results[race_id]
                race_bets = bets[bets.race_id.eq(race_id)]
                allowed = withdrawals_without_exposure(actual, predictions[predictions.race_id.eq(race_id)], race_bets)
                # 主方式では関係のない取消でも、補正方式の買い目で必要なら公式返還を必須にする。
                validated_result_status(actual, allowed)
                payouts[race_id] = actual.attrs.get('official_payouts', {})
                refunds[race_id] = actual.attrs.get('official_refunds', {})
            judged = compare_recommended_bets(bets, payouts, refunds)
            total = len(judged)
            known = judged.bet_result.ne('未確認')
            settled = int(known.sum())
            totals = import_module('src.00_common.purchase_plan').return_summary(judged)
            payout = totals['payout']
            complete = settled == total and not failures
            result = {'status': 'no_bets' if total == 0 else ('completed' if complete else 'pending'),
                      'total_bets': total, 'settled_bets': settled,
                      'hit_bets': int(judged.bet_result.eq('的中').sum()),
                      'refund_bets': int(judged.bet_result.eq('返還').sum()),
                      'purchase': totals['purchase'], 'payout': payout,
                      'recovery_rate': totals['recovery_rate'] if complete else None,
                      'missing_races': failures}
            target = output / method
            target.mkdir(parents=True, exist_ok=True)
            judged.to_csv(target / 'bets_results.csv', index=False, encoding='utf-8-sig')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result = {'status': 'incomplete', 'reason': str(exc)}
        results[method] = result
    output.mkdir(parents=True, exist_ok=True)
    (output / 'returns.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return results
