"""Frozen raw-probability ticket-policy experiment, separate from calibration."""
from __future__ import annotations

import hashlib
from importlib import import_module
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

CATEGORY = 'ticket_selection'
POLICY = 'ev_cap2'
DEFINITION = {'version': 1, 'calibration_method': 'none',
              'minimum_recovery_rate_percent': 110, 'maximum_recovery_rate_percent': 200,
              'ranking': ['recommendation_score', 'recovery_rate_percent'],
              'maximum_tickets_per_race': 3, 'stake_yen': 100,
              'candidate_universe': 'existing_eligible_candidates'}
POLICY_ID = 'policy_' + hashlib.sha256(json.dumps(DEFINITION, sort_keys=True).encode()).hexdigest()[:20]
FILES = ('predictions.csv', 'bets.csv', 'recommended_bets.csv', 'definition.json')


def select_candidates(candidates):
    """Apply the cap before selecting up to three tickets; preserve live ranking."""
    eligible = candidates[candidates.recovery_rate_percent.between(110, 200)].copy()
    eligible['selection_policy'] = POLICY
    eligible['selection_policy_id'] = POLICY_ID
    plan = import_module('src.00_common.purchase_plan').select_plan(eligible)
    return eligible, plan


def save(predictions, output: Path, odds):
    root = Path(output) / CATEGORY
    root.mkdir(parents=True, exist_ok=True)
    status = {'status': 'unavailable', 'selection_policy_id': POLICY_ID,
              'reason': '補正なしの予想または取得オッズがありません。'}
    cal = import_module('src.40_ai_modeling.common.calibration')
    raw = next((frame for method, frame in cal.comparison_predictions(predictions) if method == 'none'), None)
    if (raw is not None and odds is not None and not odds.empty
            and set(raw.race_id.astype(str)) <= set(odds.race_id.astype(str))):
        raw = raw.assign(selection_policy=POLICY, selection_policy_id=POLICY_ID)
        candidates = import_module('src.40_ai_modeling.betting').calculate_bet_recommendations(raw, odds, best_only=False)
        candidates['calibration_method'] = 'none'
        candidates['calibration_id'] = raw.calibration_id.iloc[0]
        candidates, plan = select_candidates(candidates)
        directory = root / POLICY
        directory.mkdir(parents=True, exist_ok=True)
        raw.to_csv(directory / 'predictions.csv', index=False, encoding='utf-8-sig')
        candidates.to_csv(directory / 'bets.csv', index=False, encoding='utf-8-sig')
        plan.to_csv(directory / 'recommended_bets.csv', index=False, encoding='utf-8-sig')
        (directory / 'definition.json').write_text(json.dumps(DEFINITION, indent=2), encoding='utf-8')
        status = {'status': 'ok', 'selection_policy_id': POLICY_ID,
                  'recommended_rows': len(plan), 'races_available': raw.race_id.nunique(),
                  'races_expected': raw.race_id.nunique()}
    (root / 'status.json').write_text(json.dumps({POLICY: status}, ensure_ascii=False, indent=2), encoding='utf-8')


def aggregate(output: Path, race_directories):
    directories = list(race_directories)
    if not directories or not any((d / CATEGORY / 'status.json').exists() for d in directories):
        return
    sources = [d / CATEGORY / POLICY for d in directories]
    valid = []
    for directory in sources:
        try:
            status = json.loads((directory.parent / 'status.json').read_text(encoding='utf-8'))[POLICY]
            valid.append(status['status'] == 'ok' and status['selection_policy_id'] == POLICY_ID
                         and all((directory / name).is_file() for name in FILES)
                         and json.loads((directory / 'definition.json').read_text(encoding='utf-8')) == DEFINITION)
        except (OSError, ValueError, KeyError, TypeError):
            valid.append(False)
    root = Path(output) / CATEGORY
    root.mkdir(parents=True, exist_ok=True)
    status = {'status': 'ok' if all(valid) else 'incomplete', 'selection_policy_id': POLICY_ID,
              'races_available': sum(valid), 'races_expected': len(directories)}
    if all(valid):
        target = root / POLICY
        target.mkdir(parents=True, exist_ok=True)
        for name in FILES[:3]:
            frames = [pd.read_csv(d / name, dtype={'race_id': str, 'horse_id': str, 'selection': str}) for d in sources]
            pd.concat(frames, ignore_index=True).to_csv(target / name, index=False, encoding='utf-8-sig')
            if name == 'recommended_bets.csv':
                status['recommended_rows'] = sum(len(f) for f in frames)
        shutil.copy2(sources[0] / 'definition.json', target / 'definition.json')
    (root / 'status.json').write_text(json.dumps({POLICY: status}, ensure_ascii=False, indent=2), encoding='utf-8')


def _relative_paths(day):
    return [f'{CATEGORY}/{day}/status.json',
            *(f'{CATEGORY}/{day}/{POLICY}/{name}' for name in FILES)]


def snapshot_inputs(predictions_root: Path, day: str, output: Path):
    digest = import_module('src.50_result_comparison.calibration_comparison').digest
    hashes = {}
    for name in _relative_paths(day):
        source = predictions_root / name
        hashes[name] = digest(source)
        if hashes[name] is not None:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return hashes


def validate_snapshot(predictions_root: Path, day: str, hashes):
    digest = import_module('src.50_result_comparison.calibration_comparison').digest
    if set(hashes) != set(_relative_paths(day)):
        raise ValueError('選定検証の入力ハッシュが不正です。')
    if any(digest(predictions_root / name) != value for name, value in hashes.items()):
        raise ValueError('選定検証の買い目が照合開始後に変更されました。')


def settle(snapshot: Path, day: str, public_predictions, actual_results, output: Path):
    def validate(method, status, predictions, bets):
        definition = snapshot / CATEGORY / day / method / 'definition.json'
        if (status.get('selection_policy_id') != POLICY_ID
                or json.loads(definition.read_text(encoding='utf-8')) != DEFINITION):
            raise ValueError('選定条件の定義が一致しません。')
        if status.get('recommended_rows') != len(bets):
            raise ValueError('保存した購入点数と検証データが一致しません。')
        for frame in (predictions, bets):
            if (not {'selection_policy', 'selection_policy_id'} <= set(frame)
                    or not frame.selection_policy.eq(POLICY).all() or not frame.selection_policy_id.eq(POLICY_ID).all()):
                raise ValueError('選定条件の識別子が一致しません。')
        if (not {'stake_yen', 'recovery_rate_percent', 'estimated_probability', 'odds_used'} <= set(bets)
                or not bets.stake_yen.eq(100).all() or not bets.recovery_rate_percent.between(110, 200).all()
                or (not bets.empty and not np.isclose(bets.recovery_rate_percent,
                                  bets.estimated_probability * bets.odds_used * 100).all())):
            raise ValueError('検証買い目の金額または推定回収率が不正です。')
    results = import_module('src.50_result_comparison.calibration_comparison').settle_saved_comparisons(
        snapshot, day, public_predictions, actual_results, output,
        category=CATEGORY, methods=(POLICY,), calibration_methods={POLICY: 'none'}, validate_method=validate)
    # Empty plans remain distinguishable from unavailable/missing predictions.
    results[POLICY]['selection_policy_id'] = POLICY_ID
    (output / 'returns.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return results
