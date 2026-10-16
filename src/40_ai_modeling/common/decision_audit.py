"""予想時点の入力・設定・採否を保存し、後日の比較に使う監査アーカイブを作る。"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def audit_odds(odds, output, root):
    """DB登録前に取消等で外した行も結合する。予想入力の順序は維持する。"""
    inputs = odds.copy()
    if inputs.empty:
        for column in ('race_id', 'bet_type', 'selection_1', 'selection_2', 'selection_3', 'odds_min'):
            if column not in inputs:
                inputs[column] = pd.Series(dtype=object)
    inputs['prediction_input'] = True
    raw = Path(output) / 'fetch' / 'odds.csv'
    if raw.exists():
        acquired = pd.read_csv(raw, dtype={'race_id': str})

        def keys(frame):
            return list(zip(frame.race_id.astype(str), frame.bet_type,
                            *(pd.to_numeric(frame[c], errors='coerce').fillna(-1) for c in ('selection_1', 'selection_2', 'selection_3'))))

        known = set(keys(inputs))
        excluded = acquired.loc[[key not in known for key in keys(acquired)]].copy()
        excluded['prediction_input'] = False
        inputs = pd.concat([inputs, excluded], ignore_index=True)
    content = inputs.to_csv(index=False, lineterminator='\n').encode('utf-8-sig')
    (root / 'audit_odds.csv').write_bytes(content)
    return inputs, sha256(content).hexdigest()


def start_snapshot(root: Path, odds: pd.DataFrame):
    """実際に計算へ渡すオッズのバイト列を固定し、3方式で同じIDを使う。"""
    betting = import_module('src.40_ai_modeling.betting')
    purchase = import_module('src.00_common.purchase_plan')
    content = odds.to_csv(index=False, lineterminator='\n').encode('utf-8-sig')
    digest = sha256(content).hexdigest()
    snapshot_id = 'odds_' + digest
    policy = {
        'version': 1, 'minimum_hit_probability': betting.MIN_HIT_PROBABILITY,
        'minimum_recovery_rate_percent': betting.MIN_RECOVERY_RATE_PERCENT,
        'bet_type_reliability': betting.BET_TYPE_RELIABILITY,
        'primary_bet_types': sorted(betting.PRIMARY_BET_TYPES),
        'recommended_limit_per_race': purchase.LIMIT, 'stake_per_ticket': purchase.STAKE,
        'ranking': ['recommendation_score descending', 'recovery_rate_percent descending', 'primary bet type for ties', 'stable input order'],
        'odds_rule': 'odds_min',
        'probability_algorithm_sha256': sha256(Path(betting.__file__).read_bytes()).hexdigest(),
        'purchase_algorithm_sha256': sha256(Path(purchase.__file__).read_bytes()).hexdigest(),
    }
    policy_id = 'policy_' + sha256(_json(policy).encode('utf-8')).hexdigest()
    context = {'format_version': 1, 'odds_snapshot_id': snapshot_id,
               'odds_sha256': digest, 'decision_at': datetime.now(timezone.utc).isoformat(),
               'policy_id': policy_id, 'policy': policy,
               'odds_rows': len(odds),
               'acquisition_time_status': 'known' if 'fetched_at' in odds and odds.fetched_at.notna().all() and len(odds) else 'unknown_or_partial'}
    (root / 'odds.csv').write_bytes(content)
    (root / 'snapshot.json').write_text(_json(context), encoding='utf-8')
    return context


def save_decisions(directory, frame, odds, context, method):
    """閾値適用前の全入力行を評価する。未計算値は欠損のまま保存する。"""
    betting = import_module('src.40_ai_modeling.betting')
    decisions = betting.calculate_bet_recommendations(frame, odds, best_only=False, include_rejected=True)
    ordered_odds = pd.concat([group for _, group in odds.groupby('race_id', sort=False)], ignore_index=True) if len(odds) else odds
    for column in ('fetched_at', 'source_url', 'source_cname', 'source_sha256', 'source_html', 'odds_max'):
        if column in ordered_odds:
            decisions[column] = ordered_odds[column].to_numpy()
    decisions['odds_row_index'] = [index for _, group in odds.reset_index(drop=True).groupby('race_id', sort=False) for index in group.index]
    for column in ('odds_snapshot_id', 'policy_id', 'decision_at'):
        decisions[column] = context[column]
    decisions['calibration_method'] = method
    decisions['calibration_id'] = frame.calibration_id.iloc[0]
    decisions['model_run_id'] = frame.model_run_id.iloc[0] if 'model_run_id' in frame else ''
    # 行番号で元オッズへ戻れる。重複した買い目も個別の評価行として残す。
    decisions['audit_odds_sha256'] = context['audit_odds_sha256']
    decisions['candidate_id'] = [f'audit_{context["audit_odds_sha256"]}:{i}' for i in decisions.odds_row_index]
    decisions.to_csv(directory / 'decisions.csv', index=False, encoding='utf-8-sig')


def archive_snapshot(root: Path, output: Path, context):
    """監査一式を圧縮して公開保存対象へ渡す。raw HTMLも取消・非数値表示の根拠になる。"""
    archive_dir = root / 'archives'
    archive_dir.mkdir(exist_ok=True)
    # 同じオッズでも判断時刻・モデルが違えば別アーカイブとして保持する。
    archive_id = sha256((_json(context) + (root / 'status.json').read_text(encoding='utf-8')).encode('utf-8')).hexdigest()
    archive = archive_dir / f'{archive_id}.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as bundle:
        betting = import_module('src.40_ai_modeling.betting')
        bundle.writestr('implementation/betting.py', Path(betting.__file__).read_bytes())
        purchase = import_module('src.00_common.purchase_plan')
        bundle.writestr('implementation/purchase_plan.py', Path(purchase.__file__).read_bytes())
        for source in sorted(root.rglob('*')):
            if source.is_file() and 'archives' not in source.relative_to(root).parts and source.suffix in {'.csv', '.json'}:
                bundle.writestr(source.relative_to(root).as_posix(), source.read_bytes())
        # 日次ワークフローではDB登録前の全解析行と取得HTMLを追加で残す。
        raw_odds = Path(output) / 'fetch' / 'odds.csv'
        if raw_odds.exists():
            bundle.writestr('acquisition/odds.csv', raw_odds.read_bytes())
        raw_directory = Path(output) / 'fetch' / 'odds'
        if raw_directory.exists():
            for source in sorted(raw_directory.iterdir()):
                if source.is_file() and source.suffix in {'.html', '.json'}:
                    bundle.writestr('acquisition/html/' + source.name, source.read_bytes())
        # 各入力・判断CSVも独立に整合性確認できるようにする。
        manifest = {name: sha256(bundle.read(name)).hexdigest() for name in bundle.namelist()}
        bundle.writestr('manifest.json', _json({'format_version': 1, 'sha256': manifest}))
    return archive
