"""Compare calibration offline; never update models, source DBs or public pages.

Run from the repository root with the project's Python environment. Historical
model test predictions are repurposed as calibration development data. Published
September/October results are used only for the final retrospective evaluation.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
betting = import_module('src.40_ai_modeling.betting')
KEYS = ['race_id', 'bet_type', 'selection']
EPS = 1e-7


def probability_metrics(y, p) -> dict:
    """Use fixed bins for ECE; Brier/log loss are primary calibration measures."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(
        np.mean(bins == i) * abs(p[bins == i].mean() - y[bins == i].mean())
        for i in range(10) if (bins == i).any()
    )
    return {'rows': len(y), 'mean_probability': float(p.mean()),
            'actual_rate': float(y.mean()), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'brier': float(brier_score_loss(y, p)), 'ece_10_bins': float(ece),
            'auc': float(roc_auc_score(y, p))}


def fit_calibrator(name: str, p, y):
    """Fit without class balancing: calibration must match event frequencies."""
    if name == 'sigmoid':
        return LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000).fit(
            logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1), y)
    return IsotonicRegression(out_of_bounds='clip').fit(p, y)


def calibrate(name, model, p):
    p = np.asarray(p, dtype=float)
    if name == 'sigmoid':
        return model.predict_proba(logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1))[:, 1]
    return model.predict(p)


def read_inputs(model_dir: Path, database: Path, published: Path):
    """Reconstruct the training weight only if source counts match saved metadata."""
    meta = json.loads((model_dir / 'metrics.json').read_text(encoding='utf-8'))
    old = pd.read_parquet(model_dir / 'test_predictions.parquet')
    old['race_id'] = old.race_id.astype(str)
    old['horse_id'] = old.horse_id.astype(str)
    with duckdb.connect(str(database), read_only=True) as con:
        n, positives = con.execute(
            'SELECT count(*),sum(CASE WHEN finish_position<=3 THEN 1 ELSE 0 END) '
            'FROM race_records WHERE dataset_type=? AND finish_position>0 '
            'AND race_date BETWEEN ? AND ?', ['historical', *meta['date_ranges']['train']]
        ).fetchone()
        dates = con.execute(
            'SELECT race_id,horse_id,race_date,finish_position FROM race_records '
            'WHERE dataset_type=?', ['historical']).df()
    assert n == meta['train_rows'], 'Cannot reconstruct the original pos_weight reliably'
    assert 0 < positives < n
    weight = (n - positives) / positives
    dates['race_id'] = dates.race_id.astype(str)
    dates['horse_id'] = dates.horse_id.astype(str)
    old = old.merge(dates, on=['race_id', 'horse_id'], how='left', validate='one_to_one')
    assert old.race_date.notna().all()
    assert len(old) == meta['test_rows']
    assert np.array_equal(old.actual_top3.to_numpy(), old.finish_position.le(3).to_numpy())
    old = old.sort_values(['race_date', 'race_id', 'horse_id'], kind='stable').reset_index(drop=True)
    comparisons, candidates = [], []
    for p in sorted(published.glob('*_bets_results.csv')):
        day = p.name[:10]
        prediction = pd.read_csv(published / f'{day}_comparison.csv',
                                 dtype={'race_id': str, 'horse_id': str})
        assert prediction.model_run_id.eq(meta['model_run_id']).all()
        prediction['day'] = day
        comparisons.append(prediction)
        tickets = pd.read_csv(p, dtype={'race_id': str, 'selection': str})
        tickets['day'] = day
        candidates.append(tickets)
    recent = pd.concat(comparisons, ignore_index=True)
    bets = pd.concat(candidates, ignore_index=True)
    assert not recent.duplicated(['race_id', 'horse_id']).any()
    assert not bets.duplicated(KEYS).any()
    assert recent.result_status.isin(['finished', 'scratched', 'excluded', 'did_not_finish', 'disqualified']).all()
    assert not bets.bet_result.eq('未確認').any()
    assert bets.payout_per_100.notna().all()
    assert old.race_date.max() < pd.to_datetime(recent.race_date).min()
    return meta, old, recent, bets, int(n), int(positives), weight


def reselect(recent, candidates, probabilities):
    """Rebuild only saved candidates using original odds and current production rules.

    Missing previously filtered-out odds are NOT filled from final odds or guessed.
    Recompute compound-ticket probabilities from all horses in each race, even if
    some horses never appeared in the saved candidate tickets.
    """
    predictions = recent.copy()
    predictions['top3_probability'] = probabilities
    odds = candidates[KEYS].copy()
    parts = candidates.selection.str.split('-')
    for position in range(3):
        odds[f'selection_{position + 1}'] = parts.map(
            lambda values: int(values[position]) if len(values) > position else np.nan)
    odds['odds_min'] = candidates.odds_used.to_numpy()
    recalculated = betting.calculate_bet_recommendations(predictions, odds, best_only=False)
    # The website re-sorts by score, not by the primary-bet-type flag. Match it.
    selected = recalculated.sort_values(
        ['recommendation_score', 'recovery_rate_percent'], ascending=False, kind='stable'
    ).groupby('race_id', sort=False).head(3)
    selected = selected.merge(
        candidates[KEYS + ['day', 'bet_result', 'payout_per_100']],
        on=KEYS, how='left', validate='one_to_one')
    assert selected.payout_per_100.notna().all()
    return selected, recalculated


def ticket_summary(selected, race_count):
    """Equal 100-yen stakes; no-bet races contribute neither stake nor payout."""
    stake = len(selected) * 100
    payout = float(selected.payout_per_100.sum())
    return {'tickets': len(selected), 'races_bet': int(selected.race_id.nunique()),
            'races_skipped': race_count - int(selected.race_id.nunique()),
            'stake_yen': stake, 'payout_yen': payout, 'profit_yen': payout - stake,
            'roi_percent': payout / stake * 100 if stake else None,
            'hit_rate': float(selected.bet_result.eq('的中').mean()) if stake else None,
            'mean_estimated_roi': float(selected.recovery_rate_percent.mean()) if stake else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/top3/model_20260829_214945_ef2111')
    parser.add_argument('--database', type=Path, default=ROOT / 'data/racing.duckdb')
    parser.add_argument('--published', type=Path, default=ROOT / 'docs/predictions')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/analysis/probability_calibration')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta, old, recent, candidates, n, positives, weight = read_inputs(args.model_dir, args.database, args.published)
    # Choose sigmoid vs isotonic on a date-held-out suffix of OLD predictions.
    # No September/October labels enter fitting or method selection.
    dates = sorted(old.race_date.unique())
    boundary = dates[int(len(dates) * .7)]
    fit = old[old.race_date < boundary]
    holdout = old[old.race_date >= boundary]
    p_fit = fit.predicted_probability.to_numpy(dtype=float)
    p_holdout = holdout.predicted_probability.to_numpy(dtype=float)
    development = []
    for name in ['sigmoid', 'isotonic']:
        model = fit_calibrator(name, p_fit, fit.actual_top3.to_numpy(dtype=int))
        pred = calibrate(name, model, p_holdout)
        development.append({'method': name, **probability_metrics(holdout.actual_top3, pred)})
    chosen = min(development, key=lambda row: row['log_loss'])['method']
    final_model = fit_calibrator(chosen, old.predicted_probability.to_numpy(dtype=float), old.actual_top3.to_numpy(dtype=int))
    q = recent.top3_probability.to_numpy(dtype=float)
    variants = {
        'raw': q,
        # Ideal weighted-BCE correction: subtract log(pos_weight) from logits.
        # This is an analytical diagnostic, not a claim the fitted MLP is ideal.
        'weight_corrected': expit(logit(np.clip(q, EPS, 1 - EPS)) - np.log(weight)),
        'learned_' + chosen: calibrate(chosen, final_model, q),
    }
    y = (recent.result_status.eq('finished') & pd.to_numeric(recent.finish_position, errors='coerce').le(3)).astype(int)
    metrics, betting_results, daily, curves, outputs, ticket_groups = [], [], [], [], [], []
    races = recent.race_id.nunique()
    saved_displayed = candidates.sort_values(
        ['recommendation_score', 'recovery_rate_percent'], ascending=False, kind='stable'
    ).groupby('race_id', sort=False).head(3)
    for name, p in variants.items():
        metrics.append({'method': name, **probability_metrics(y, p),
                        'mean_probability_sum_per_race': float(pd.Series(p).groupby(recent.race_id).sum().mean())})
        selected, rescored = reselect(recent, candidates, p)
        if name == 'raw':
            # Regression check: the replay must recover the actual saved baseline.
            assert set(map(tuple, selected[KEYS].to_numpy())) == set(map(tuple, saved_displayed[KEYS].to_numpy()))
            assert np.isclose(selected.payout_per_100.sum(), saved_displayed.payout_per_100.sum())
            check = rescored.merge(candidates, on=KEYS, suffixes=('_new', '_old'), validate='one_to_one')
            assert len(check) == len(candidates)
            assert np.allclose(check.recovery_rate_percent_new, check.recovery_rate_percent_old)
        betting_results.append({'method': name, **ticket_summary(selected, races)})
        for bet_type, group in selected.groupby('bet_type'):
            ticket_groups.append({'method': name, 'bet_type': bet_type,
                                  'tickets': len(group),
                                  'predicted_hit_rate': float(group.estimated_probability.mean()),
                                  'actual_hit_rate': float(group.bet_result.eq('的中').mean()),
                                  'roi_percent': float(group.payout_per_100.sum() / len(group))})
        selected.assign(method=name).to_csv(args.output / f'tickets_{name}.csv', index=False, encoding='utf-8-sig')
        for day in sorted(recent.day.unique()):
            subset = selected[selected.day.eq(day)]
            daily.append({'method': name, 'day': day, **ticket_summary(subset, recent.loc[recent.day.eq(day), 'race_id'].nunique())})
        bins = np.minimum((p * 10).astype(int), 9)
        for i in range(10):
            mask = bins == i
            if mask.any():
                curves.append({'method': name, 'bin_lower': i / 10, 'bin_upper': (i + 1) / 10,
                               'rows': int(mask.sum()), 'predicted': float(p[mask].mean()), 'actual': float(y[mask].mean())})
        output = recent[['day', 'race_id', 'horse_id', 'horse_number']].copy()
        output['method'], output['probability'], output['actual_top3'] = name, p, y
        outputs.append(output)
    pd.DataFrame(development).to_csv(args.output / 'calibrator_selection.csv', index=False)
    pd.DataFrame(metrics).to_csv(args.output / 'probability_metrics.csv', index=False)
    pd.DataFrame(betting_results).to_csv(args.output / 'betting_metrics.csv', index=False)
    pd.DataFrame(daily).to_csv(args.output / 'daily_returns.csv', index=False)
    pd.DataFrame(curves).to_csv(args.output / 'reliability_bins.csv', index=False)
    pd.DataFrame(ticket_groups).to_csv(args.output / 'selected_ticket_calibration.csv', index=False)
    pd.concat(outputs, ignore_index=True).to_csv(args.output / 'horse_probabilities.csv', index=False)
    details = {
        'model_id': meta['model_run_id'], 'train_rows': n, 'train_positives': positives,
        'pos_weight': weight, 'calibration_rows': len(old),
        'calibration_period': [str(old.race_date.min().date()), str(old.race_date.max().date())],
        'selection_fit_rows': len(fit), 'selection_holdout_rows': len(holdout),
        'selection_holdout_start': str(pd.Timestamp(boundary).date()),
        'selected_calibrator': chosen, 'development': development,
        'evaluation_period': [recent.day.min(), recent.day.max()],
        'evaluation_days': recent.day.nunique(), 'evaluation_races': int(races),
        'candidate_tickets': len(candidates), 'metrics': metrics, 'betting': betting_results,
        'selected_ticket_groups': ticket_groups,
        'limitations': ['Previously published results have already been inspected; not a blind prospective test.',
                        'Reselection uses only previously saved candidates, not the complete original odds universe.',
                        'Ticket outcomes and odds are frozen; no final odds used to generate decisions.',
                        'Historical model test predictions are now calibration development data, not an untouched test set.',
                        'Nine race days cannot establish stable positive expected returns.'],
    }
    if chosen == 'sigmoid':
        details['calibrator'] = {'slope': float(final_model.coef_[0, 0]), 'intercept': float(final_model.intercept_[0])}
    else:
        details['calibrator'] = {'x': final_model.X_thresholds_.tolist(), 'y': final_model.y_thresholds_.tolist()}
    (args.output / 'summary.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    labels = {'raw': '補正なし', 'weight_corrected': '重み付け補正', 'learned_' + chosen: '学習による確率校正'}
    report = [
        '# 確率補正の比較結果', '',
        '本番モデル・バンドル・公開ページは更新していない。保存済みデータによるオフライン分析。', '',
        '## データと手順', '',
        f'- 対象モデル: `{meta["model_run_id"]}`',
        f'- 元の学習件数: {n:,}頭、3着内: {positives:,}頭。登録指標の学習件数との一致を確認。',
        f'- 復元した正例重み: {weight:.9f}（負例数÷正例数）。',
        f'- 補正開発データ: {details["calibration_period"][0]}～{details["calibration_period"][1]}、{len(old):,}頭。',
        f'- 前半{len(fit):,}頭で補正を学習、{details["selection_holdout_start"]}以降{len(holdout):,}頭のLog lossでsigmoidとisotonicを比較。',
        f'- 選択された方式: {chosen}。方式決定後、過去予測{len(old):,}頭で補正器を再学習。',
        f'- 最終比較: {recent.day.min()}～{recent.day.max()}、{recent.day.nunique()}開催日・{races}レース・{len(recent):,}頭。',
        '- 9～10月の正解ラベルは補正学習や方式選択へ使用していない。', '',
        '## 確率評価', '',
        'Log loss・Brier・ECEは小さいほど良い。ECEは幅0.1の固定10区間。', '',
        '| 方式 | 平均予測確率 | 実3着内率 | Log loss | Brier | ECE | AUC | レース内確率合計の平均 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in metrics:
        report.append(f'| {labels[row["method"]]} | {row["mean_probability"]:.2%} | {row["actual_rate"]:.2%} | '
                      f'{row["log_loss"]:.5f} | {row["brier"]:.5f} | {row["ece_10_bins"]:.5f} | '
                      f'{row["auc"]:.5f} | {row["mean_probability_sum_per_race"]:.3f} |')
    report += ['', '## 保存候補内での買い目再選定', '',
               '当時の取得オッズ・券種別閾値・信頼度係数・公開側の並べ替え・上位3点・各100円を固定。',
               '確率を変更した後、組合せ券種の確率と期待値も再計算して条件に合う候補を選び直した。',
               '補正なしの買い目キー・払戻と、保存された実際の掲載上位3点との一致を検査済み。', '',
               '| 方式 | 購入点数 | 購入レース | 見送り | 購入額 | 払戻額 | 損益 | 回収率 |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in betting_results:
        report.append(f'| {labels[row["method"]]} | {row["tickets"]} | {row["races_bet"]} | {row["races_skipped"]} | '
                      f'{row["stake_yen"]:,}円 | {row["payout_yen"]:,.0f}円 | {row["profit_yen"]:+,.0f}円 | {row["roi_percent"]:.2f}% |')
    report += ['', '## 解釈と制約', '',
               '- 重み付け補正は `sigmoid(logit(q) - log(pos_weight))`。理想的な重み付き損失の補正式であり、完全な補正を保証しない。',
               '- 全馬の確率評価は大幅に改善したが、単調変換なので順位とAUCは変わらない。',
               '- 選ばれた買い目の校正は全馬の校正と別問題。券種別の推定的中率・実的中率はselected_ticket_calibration.csvを参照。',
               '- 当初の候補から除外された買い目の全オッズがないため、2,828件の保存候補内の再選定であり、完全な戦略バックテストではない。',
               '- 元と同じ買い目を同じ金額で買い続けるなら、確率表示を変えるだけでは払戻・回収率は変わらない。',
               '- 元のモデルテスト予測を校正開発へ再利用した。その期間を独立テストとして再度主張しない。',
               '- 9～10月の結果は分析前から閲覧済みで、完全な盲検・将来検証ではない。9開催日だけで安定した利益を判断しない。',
               '- 元データ・重み・予想・払戻は読み取りのみ。新規出力はこの分析ディレクトリに限定。', '',
               '再実行: `.venv/Scripts/python.exe scripts/compare_probability_calibration.py`', '']
    (args.output / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
