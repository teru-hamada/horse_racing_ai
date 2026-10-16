"""Offline ticket calibration and chronological policy audit.

Only the old saved candidate universe is available. Do not interpret these
results as an exhaustive odds backtest or an untouched prospective experiment.
No production model, betting policy, database or public page is modified.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
betting = import_module('src.40_ai_modeling.betting')
KEYS = ['race_id', 'bet_type', 'selection']


def score_saved_candidates(source: Path, calibration: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute all OLD candidates, including ones the corrected rules exclude."""
    probabilities = pd.read_csv(calibration / 'horse_probabilities.csv', dtype={'race_id': str, 'horse_id': str})
    probabilities = probabilities[probabilities.method.eq('learned_sigmoid')]
    horses, tickets = [], []
    for path in sorted(source.glob('*_bets_results.csv')):
        day = path.name[:10]
        frame = pd.read_csv(source / f'{day}_comparison.csv', dtype={'race_id': str, 'horse_id': str})
        frame['day'] = day
        horses.append(frame)
        frame = pd.read_csv(path, dtype={'race_id': str, 'selection': str})
        frame['day'] = day
        tickets.append(frame)
    horses = pd.concat(horses, ignore_index=True).merge(
        probabilities[['race_id', 'horse_id', 'probability']],
        on=['race_id', 'horse_id'], how='left', validate='one_to_one')
    assert horses.probability.notna().all()
    tickets = pd.concat(tickets, ignore_index=True)
    assert not tickets.duplicated(KEYS).any()
    assert tickets.bet_result.isin(['的中', '不的中', '返還']).all()
    # Refunds remain 100-yen returns in ROI, but are neither wins nor losses
    # for probability fitting/evaluation. Never label them as losing tickets.
    groups = {rid: g for rid, g in horses.groupby('race_id', sort=False)}
    values = []
    for row in tickets.itertuples(index=False):
        race = groups[row.race_id]
        top3 = dict(zip(race.horse_number.astype(int), race.probability))
        total = sum(top3.values())
        weights = {n: p / total for n, p in top3.items()}
        selections = tuple(map(int, row.selection.split('-')))
        if row.bet_type == 'bracket_quinella':
            from itertools import permutations
            frames = dict(zip(race.horse_number.astype(int), race.frame_number.astype(int)))
            p = sum(betting._ordered_probability(order, weights)
                    for order in permutations(weights, 2)
                    if tuple(sorted(frames[n] for n in order)) == tuple(sorted(selections)))
        else:
            p, _ = betting._estimated_probability(row.bet_type, selections, weights, top3)
        values.append(p)
    tickets['p'] = values
    tickets['hit'] = tickets.bet_result.eq('的中').astype(int)
    tickets['ev'] = tickets.p * tickets.odds_used
    tickets['score'] = tickets.ev * tickets.bet_type.map(betting.BET_TYPE_RELIABILITY)
    tickets['eligible'] = ((tickets.p >= tickets.bet_type.map(betting.MIN_HIT_PROBABILITY))
                           & (tickets.ev >= betting.MIN_RECOVERY_RATE_PERCENT / 100))
    assert np.isfinite(tickets[['p', 'ev', 'odds_used']].to_numpy()).all()
    assert tickets.p.between(0, 1).all()
    return tickets, horses


def select(tickets, spec):
    """Each fixed policy uses only decision-time p, odds and ticket type."""
    p_column = 'ticket_p' if spec.get('ticket_calibrated') else 'p'
    p = tickets[p_column]
    ev = p * tickets.odds_used
    allowed = spec.get('types', set(betting.BET_TYPE_LABELS))
    mask = (tickets.bet_type.isin(allowed)
            & (p >= tickets.bet_type.map(betting.MIN_HIT_PROBABILITY))
            & (p >= spec.get('min_p', 0))
            & (ev >= spec.get('min_ev', 1.1))
            & (ev <= spec.get('max_ev', np.inf))
            & (tickets.odds_used <= spec.get('max_odds', np.inf)))
    chosen = tickets[mask].copy()
    chosen['decision_p'] = p[mask]
    chosen['decision_ev'] = ev[mask]
    chosen['decision_score'] = chosen.decision_ev
    if not spec.get('ev_rank'):
        chosen['decision_score'] *= chosen.bet_type.map(betting.BET_TYPE_RELIABILITY)
    return chosen.sort_values(['decision_score', 'decision_ev'], ascending=False, kind='stable').groupby(
        'race_id', sort=False).head(spec.get('points', 3))


def performance(chosen, all_races):
    stake = len(chosen) * 100
    payout = float(chosen.payout_per_100.sum())
    outcomes = chosen[~chosen.bet_result.eq('返還')]
    return {'tickets': len(chosen), 'races_bet': int(chosen.race_id.nunique()),
            'skipped': all_races - int(chosen.race_id.nunique()),
            'days_bet': int(chosen.day.nunique()), 'stake': stake, 'payout': payout,
            'profit': payout - stake, 'roi': payout / stake * 100 if stake else None,
            'predicted_hit': float(outcomes.decision_p.mean()) if len(outcomes) else None,
            'actual_hit': float(outcomes.hit.mean()) if len(outcomes) else None,
            'roi_excluding_largest_payout': (payout - chosen.payout_per_100.max()) / stake * 100 if stake else None}


def diagnostic(groups, label):
    result = []
    for key, group in groups:
        outcomes = group[~group.bet_result.eq('返還')]
        if outcomes.empty:
            continue
        p, y = outcomes.p.to_numpy(), outcomes.hit.to_numpy()
        result.append({'grouping': label, 'group': str(key), 'tickets': len(group),
                       'calibration_rows': len(outcomes), 'refunds': len(group) - len(outcomes),
                       'races': int(group.race_id.nunique()), 'hits': int(y.sum()),
                       'predicted_hit': float(p.mean()), 'actual_hit': float(y.mean()),
                       'overstatement_pp': float((p.mean() - y.mean()) * 100),
                       'brier': float(brier_score_loss(y, p)),
                       'log_loss': float(log_loss(y, p, labels=[0, 1])),
                       'mean_odds': float(group.odds_used.mean()), 'mean_ev': float(group.ev.mean()),
                       'roi': float(group.payout_per_100.sum() / len(group))})
    return result


def ticket_features(tickets):
    """Fixed shared slopes and regularized type intercepts; no return/label features."""
    x = pd.DataFrame({'logit_p': logit(np.clip(tickets.p.to_numpy(), 1e-7, 1 - 1e-7)),
                      'log_odds': np.log(tickets.odds_used.to_numpy())}, index=tickets.index)
    for kind in sorted(betting.BET_TYPE_LABELS):
        x['type_' + kind] = tickets.bet_type.eq(kind).astype(float)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'docs/predictions')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'data/analysis/probability_calibration')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/analysis/ticket_selection')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tickets, horses = score_saved_candidates(args.source, args.calibration)
    days = sorted(tickets.day.unique())
    assert len(days) == 9, 'Review the predeclared chronological split for new datasets'
    dev_days, test_days = days[:4], days[4:]
    dev = tickets[tickets.day.isin(dev_days)].copy()
    test = tickets[tickets.day.isin(test_days)].copy()
    assert max(dev_days) < min(test_days)
    # Fix regularization and predictors before seeing October policy results.
    # Do not train on selected October tickets or adjust calibration to returns.
    dev_fit = dev[~dev.bet_result.eq('返還')]
    meta_model = LogisticRegression(C=.1, max_iter=2000, solver='lbfgs').fit(ticket_features(dev_fit), dev_fit.hit)
    tickets['ticket_p'] = meta_model.predict_proba(ticket_features(tickets))[:, 1]
    # Control: determine whether improvement is simply recovering market odds,
    # rather than evidence that the horse model adds information beyond odds.
    odds_features = [name for name in ticket_features(tickets).columns if name != 'logit_p']
    odds_model = LogisticRegression(C=.1, max_iter=2000, solver='lbfgs').fit(
        ticket_features(dev_fit)[odds_features], dev_fit.hit)
    tickets['odds_only_p'] = odds_model.predict_proba(ticket_features(tickets)[odds_features])[:, 1]
    # Explain selection bias at three stages, using calibrated horse probabilities.
    stages = {'all_saved_candidates': tickets, 'eligible_candidates': tickets[tickets.eligible],
              'displayed_top3': select(tickets, {})}
    baseline = pd.read_csv(args.calibration / 'tickets_learned_sigmoid.csv', dtype={'race_id': str, 'selection': str})
    assert set(map(tuple, stages['displayed_top3'][KEYS].to_numpy())) == set(map(tuple, baseline[KEYS].to_numpy()))
    diagnostics = []
    for stage, stage_frame in stages.items():
        for partition, part_days in [('all', days), ('development', dev_days), ('evaluation', test_days)]:
            frame = stage_frame[stage_frame.day.isin(part_days)]
            rows = diagnostic(frame.groupby('bet_type', observed=True), 'type')
            odds_band = pd.cut(frame.odds_used, [0, 2, 5, 10, 20, 50, np.inf], right=False)
            rows += diagnostic(frame.groupby(['bet_type', odds_band], observed=True), 'type_odds')
            ev_band = pd.cut(frame.ev, [0, 1.1, 1.5, 2, 3, 5, np.inf], right=False)
            rows += diagnostic(frame.groupby(['bet_type', ev_band], observed=True), 'type_ev')
            for row in rows:
                row['stage'], row['partition'] = stage, partition
            diagnostics.extend(rows)
    pd.DataFrame(diagnostics).to_csv(args.output / 'calibration_diagnostics.csv', index=False, encoding='utf-8-sig')
    # Small, explicit hypotheses; do not search arbitrary thresholds on October.
    simple = {'win', 'place'}
    policies = {
        'baseline': {}, 'one_point': {'points': 1},
        'win_only': {'types': {'win'}}, 'place_only': {'types': {'place'}},
        'win_place': {'types': simple},
        'win_place_odds5': {'types': simple, 'max_odds': 5},
        'win_place_odds10': {'types': simple, 'max_odds': 10},
        'win_place_ev2': {'types': simple, 'max_ev': 2},
        'win_place_ev3': {'types': simple, 'max_ev': 3},
        'win_place_edge30': {'types': simple, 'min_ev': 1.3},
        'win_place_edge50': {'types': simple, 'min_ev': 1.5},
        'win_place_p40': {'types': simple, 'min_p': .4},
        'ev_rank': {'ev_rank': True},
        'ticket_calibrated': {'ticket_calibrated': True},
        'ticket_calibrated_simple': {'ticket_calibrated': True, 'types': simple},
    }
    results, selected_frames = [], []
    for policy, spec in policies.items():
        # Leakage guard: changing outcomes must never change which tickets a
        # decision-time policy selects (outcomes are only carried for scoring).
        scrambled = tickets.assign(hit=1 - tickets.hit, payout_per_100=0,
                                    bet_result='不的中')
        assert set(map(tuple, select(tickets, spec)[KEYS].to_numpy())) == set(
            map(tuple, select(scrambled, spec)[KEYS].to_numpy()))
        for partition, part_days in [('development', dev_days), ('evaluation', test_days)]:
            frame = tickets[tickets.day.isin(part_days)]
            chosen = select(frame, spec)
            n_races = horses.loc[horses.day.isin(part_days), 'race_id'].nunique()
            results.append({'policy': policy, 'partition': partition, **performance(chosen, int(n_races))})
            selected_frames.append(chosen.assign(policy=policy, partition=partition))
    result = pd.DataFrame(results)
    # A minimum sample rule is fixed in advance; choose from development only.
    eligible = result[(result.partition == 'development') & (result.tickets >= 30)
                      & (result.days_bet >= 3) & ~result.policy.str.startswith('ticket_calibrated')]
    # Ticket-level calibration has in-sample development results: exclude it
    # from policy selection, report October as a separate fixed experiment.
    chosen_policy = eligible.sort_values(['roi', 'tickets'], ascending=False).iloc[0].policy if len(eligible) else None
    result['chosen_on_development'] = result.policy.eq(chosen_policy)
    result.to_csv(args.output / 'policy_comparison.csv', index=False, encoding='utf-8-sig')
    selections = pd.concat(selected_frames, ignore_index=True)
    selections.to_csv(args.output / 'selected_tickets.csv', index=False, encoding='utf-8-sig')
    daily = []
    for policy in policies:
        for day in days:
            frame = selections[selections.policy.eq(policy) & selections.day.eq(day)]
            daily.append({'policy': policy, 'day': day,
                          **performance(frame, int(horses.loc[horses.day.eq(day), 'race_id'].nunique()))})
    pd.DataFrame(daily).to_csv(args.output / 'daily_returns.csv', index=False, encoding='utf-8-sig')
    tickets.to_csv(args.output / 'rescored_candidates.csv', index=False, encoding='utf-8-sig')
    # Compare the learned ticket calibrator out of time, without fitting to test.
    test = tickets[tickets.day.isin(test_days) & ~tickets.bet_result.eq('返還')]
    meta_metrics = []
    for name, column in [('horse_calibrated_ticket_estimate', 'p'),
                         ('odds_aware_ticket_calibration', 'ticket_p'),
                         ('odds_and_type_only_control', 'odds_only_p')]:
        meta_metrics.append({'method': name, 'rows': len(test),
                             'predicted': float(test[column].mean()), 'actual': float(test.hit.mean()),
                             'log_loss': float(log_loss(test.hit, test[column], labels=[0, 1])),
                             'brier': float(brier_score_loss(test.hit, test[column]))})
    summary = {'development_days': dev_days, 'evaluation_days': test_days,
               'development_races': int(horses[horses.day.isin(dev_days)].race_id.nunique()),
               'evaluation_races': int(horses[horses.day.isin(test_days)].race_id.nunique()),
               'chosen_policy': chosen_policy,
               'policy_count': len(policies), 'policy_results': result.to_dict('records'),
               'ticket_calibration_evaluation': meta_metrics,
               'ticket_calibration_coefficients': dict(zip(ticket_features(tickets).columns, meta_model.coef_[0])),
               'ticket_calibration_intercept': float(meta_model.intercept_[0]),
               'refund_tickets_excluded_from_calibration': int(tickets.bet_result.eq('返還').sum()),
               'limitations': ['All nine days were previously inspected: retrospective chronological audit, not blind prospective proof.',
                               'Only candidates passing the old policy were saved; no exhaustive all-odds replay.',
                               'Multiple tickets in the same race are dependent; ticket count is not independent sample count.',
                               'Policy thresholds are explicit hypotheses; October outcomes do not select the recommended policy.',
                               'Ticket calibrator development results are in-sample and excluded from policy selection.',
                               'Five evaluation days cannot establish stable profit.']}
    # pandas uses NaN for no-bet ROI; JSON encodes these explicitly as null.
    summary = json.loads(json.dumps(summary, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
    def sanitize(value):
        if isinstance(value, dict): return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list): return [sanitize(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    summary = sanitize(summary)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    type_labels = betting.BET_TYPE_LABELS
    labels = {
        'baseline': '馬単位校正後の現行条件', 'one_point': '上位1点だけ',
        'win_only': '単勝だけ', 'place_only': '複勝だけ', 'win_place': '単勝・複勝だけ',
        'win_place_odds5': '単複・オッズ5倍以下', 'win_place_odds10': '単複・オッズ10倍以下',
        'win_place_ev2': '単複・期待回収倍率1.1～2', 'win_place_ev3': '単複・期待回収倍率1.1～3',
        'win_place_edge30': '単複・期待回収率130%以上', 'win_place_edge50': '単複・期待回収率150%以上',
        'win_place_p40': '単複・的中確率40%以上', 'ev_rank': '券種係数なしの期待値順',
        'ticket_calibrated': '券種・オッズ追加校正', 'ticket_calibrated_simple': '追加校正・単複だけ',
    }
    report = ['# 選定後の確率誤差・券種別条件の検証', '',
              '本番のモデル・買い目ロジック・自動実行・公開ページは変更していない。', '',
              '## 検証設計', '',
              f'- 前半: {dev_days[0]}～{dev_days[-1]}、4開催日・{summary["development_races"]}レース。',
              f'- 後半: {test_days[0]}～{test_days[-1]}、5開催日・{summary["evaluation_races"]}レース。',
              '- 馬単位の確率は前回のsigmoid校正を固定。元の保存候補2,828点だけを再計算。',
              '- 15個の明示的な条件を比較。後半結果を使って閾値や方式を調整していない。',
              '- 前半30点以上・3開催日以上を満たす固定条件の中から、前半回収率が最大のものを選択。',
              f'- 前半で選択された条件: {labels[chosen_policy] if chosen_policy else "該当なし"}。',
              '- 追加校正は前半全保存候補のうち返還を除いて学習。入力はlogit確率・対数オッズ・券種。',
              '- 正則化C=0.1を固定。追加校正の前半成績は学習内なので、条件選択の比較対象から除外。',
              f'- 返還{summary["refund_tickets_excluded_from_calibration"]}点は勝敗校正から除外し、回収率では公式返還額を含める。',
              '- 結果ラベルと払戻額を改変しても選択買い目が変わらないことを全方式で検査済み。', '',
              '## 掲載買い目の券種別誤差（全9開催日の記述集計）', '',
              '| 券種 | 点数 | 推定的中率 | 実的中率 | 回収率 |', '|---|---:|---:|---:|---:|']
    for row in diagnostics:
        if row['stage'] == 'displayed_top3' and row['grouping'] == 'type' and row['partition'] == 'all':
            report.append(f'| {type_labels[row["group"]]} | {row["tickets"]} | {row["predicted_hit"]:.1%} | {row["actual_hit"]:.1%} | {row["roi"]:.2f}% |')
    report += ['', 'オッズ帯・推定期待値帯ごとの同じ集計はcalibration_diagnostics.csvに保存。少数セルは判断材料にしすぎない。', '',
               '## 条件比較', '',
               '以下の後半成績が時間を進めた比較。前半の最大値と後半の最大値を事後的に組み合わせない。', '',
               '| 条件 | 前半回収率 | 後半点数 | 後半購入レース | 後半回収率 | 後半損益 |',
               '|---|---:|---:|---:|---:|---:|']
    for policy in policies:
        a = next(r for r in summary['policy_results'] if r['policy'] == policy and r['partition'] == 'development')
        b = next(r for r in summary['policy_results'] if r['policy'] == policy and r['partition'] == 'evaluation')
        report.append(f'| {labels[policy]} | {a["roi"]:.2f}% | {b["tickets"]} | {b["races_bet"]} | {b["roi"]:.2f}% | {b["profit"]:+,.0f}円 |')
    report += ['', '## 券種とオッズを使う追加校正（後半の全保存候補・返還除外）', '',
               '| 方式 | 点数 | 平均推定的中率 | 実的中率 | Log loss | Brier |', '|---|---:|---:|---:|---:|---:|']
    for row in meta_metrics:
        name = {'horse_calibrated_ticket_estimate': '馬単位校正からの券種確率',
                'odds_aware_ticket_calibration': '券種・オッズ追加校正',
                'odds_and_type_only_control': 'オッズ・券種だけの対照モデル'}[row['method']]
        report.append(f'| {name} | {row["rows"]} | {row["predicted"]:.2%} | {row["actual"]:.2%} | {row["log_loss"]:.5f} | {row["brier"]:.5f} |')
    report += ['', '## 判断', '',
               '- 前半で最良の上位1点条件は、後半105点で的中なし・回収率0%。前半の好成績は再現しなかった。',
               '- 追加校正は後半の確率指標を改善したが、購入条件を通るのは2点だけ。515%は利益の安定性を示さない。',
               '- 単複の高確率限定は後半88.14%だが、後半を見た上で勝者として採用してはいけない。今後の固定仮説の候補に留める。',
               '- 券種係数を外して期待値だけで選ぶ条件は後半17.55%。高い推定期待値を追うだけでは改善しなかった。',
               '- このデータから本番へ即採用できる、安定した100%超の条件は確認できない。', '',
               '## 制約', '',
               '- 九日分は以前の分析で閲覧済み。時系列分離は守ったが、完全な盲検・将来検証ではない。',
               '- 旧方式で保存された候補だけが対象。最初に除外された買い目の優劣は検証できない。',
               '- 同じレースの買い目は相関するため、点数を独立サンプル数とみなさない。',
               '- 券種・オッズ校正は旧候補集合に条件付けた実験。全買い目へ無条件に適用できるとは限らない。',
               '- 追加校正が出した2点の払戻に大きく依存するため、十分な新規開催日で予想のみの並行検証が必要。', '',
               '再実行: `.venv/Scripts/python.exe scripts/analyze_ticket_selection.py`', '']
    (args.output / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(result[['policy', 'partition', 'tickets', 'races_bet', 'profit', 'roi', 'chosen_on_development']].to_string(index=False))
    print('TICKET_CALIBRATION', json.dumps(meta_metrics))
    print('CHOSEN_ON_DEVELOPMENT', chosen_policy)


if __name__ == '__main__':
    main()
