# 検証対象：公開前検証、他の日付の保持、同一結果の再反映で差分が出ないことを確認する。
# 不整合な成果物が既存サイトを変更しないことや、成功タスクだけの反映を検証する。
#
from importlib import import_module as _import_module
import json

import pandas as pd
import pytest

prepare_publication = _import_module('src.60_publication.publish_predictions').prepare_publication


def test_comparison_files_are_persisted_without_changing_public_html(tmp_path, results):
    calibration = _import_module('src.40_ai_modeling.common.calibration')
    site = tmp_path / 'site'
    prepare_publication(results, site)
    page = site / 'predictions/2026-09-19.html'
    original = page.read_bytes()
    calibration.save_calibrations(tmp_path / 'calibration.json', [
        calibration.ProbabilityCalibration('m'),
        calibration.ProbabilityCalibration('m', 'class_weight', {'pos_weight': 4.}),
        calibration.ProbabilityCalibration('m', 'sigmoid', {'slope': 1., 'intercept': -1.}),
    ])
    frame = pd.read_csv(results / 'predictions.csv', dtype={'race_id': str, 'horse_id': str})
    comparison = calibration.attach_probabilities(frame, frame.top3_probability, tmp_path, {'model_run_id': 'm'})
    odds = pd.DataFrame({'race_id': frame.race_id, 'bet_type': 'place', 'selection_1': 1,
                         'selection_2': float('nan'), 'selection_3': float('nan'), 'odds_min': 3.})
    calibration.save_comparison_outputs(comparison, results, odds)
    changed = prepare_publication(results, site)
    assert changed and all(name.startswith('predictions/calibration/') for name in changed)
    assert page.read_bytes() == original
    root = site / 'predictions/calibration/2026-09-19'
    assert (root / 'status.json').exists()
    for method in calibration.METHODS:
        assert (root / method / 'predictions.csv').exists()
        assert (root / method / 'recommended_bets.csv').exists()
        assert not (root / method / 'predictions.parquet').exists()
    assert not (root / 'odds.csv').exists()
    assert prepare_publication(results, site) == []


# 結果取得・精算に使う固定の応答を作り、対象ケースを再現する。
@pytest.fixture
def results(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    row = {"race_id": "202606040510", "status": "ok", "runners": 1,
           "prediction_rows": 1, "bet_rows": 0, "model_id": "m"}
    row.update({stage: "ok" for stage in ("card", "jra_odds", "database", "prediction", "odds_match", "betting")})
    report = {"race_date": "2026-09-19", "status": "ok", "race_ids": [row["race_id"]], "races": [row]}
    (output / "date_report.json").write_text(json.dumps(report), encoding="utf-8")
    pd.DataFrame([{
        "race_id": row["race_id"], "race_date": report["race_date"], "horse_id": "h",
        "horse_number": 1, "model_run_id": "m", "top3_probability": 0.5,
        "course_name": "中山", "race_number": 10, "race_name": "Test", "horse_name": "Horse",
        "jockey_name": "Jockey", "prediction_rank": 1, "odds": 3.0, "expected_value_index": 1.5,
    }]).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(columns=["race_id", "bet_type", "selection"]).to_csv(output / "bets.csv", index=False)
    return output


# 対象外の日付を保持し、同じ予想の再公開では生成日時だけでファイルが変わらない。
def test_preserves_other_dates_and_identical_results_do_not_change_files(tmp_path, results):
    site = tmp_path / "docs"
    (site / "predictions").mkdir(parents=True)
    previous = site / "predictions/2026-09-18.html"
    previous.write_bytes(b"previous date")
    assert "predictions/2026-09-19.html" in prepare_publication(results, site)
    assert previous.read_bytes() == b"previous date"
    assert "2026-09-18.html" in (site / "index.html").read_text(encoding="utf-8")
    page = site / "predictions/2026-09-19.html"
    import re
    page.write_text(re.sub(r'<p class="meta">[^<]*</p>', '<p class="meta">生成日時: OLD TIME</p>', page.read_text(encoding="utf-8")), encoding="utf-8")
    snapshot = {str(p.relative_to(site)): p.read_bytes() for p in site.rglob("*") if p.is_file()}
    assert prepare_publication(results, site) == []
    assert snapshot == {str(p.relative_to(site)): p.read_bytes() for p in site.rglob("*") if p.is_file()}


def test_audit_archive_is_published_and_old_snapshots_are_preserved(tmp_path, results):
    root = results / 'calibration/archives'
    root.mkdir(parents=True)
    (root / 'first.zip').write_bytes(b'first snapshot')
    site = tmp_path / 'docs'
    prepare_publication(results, site)
    published = site / 'predictions/calibration/2026-09-19/archives'
    assert (published / 'first.zip').read_bytes() == b'first snapshot'
    (root / 'first.zip').unlink()
    (root / 'second.zip').write_bytes(b'second snapshot')
    prepare_publication(results, site)
    assert (published / 'first.zip').read_bytes() == b'first snapshot'
    assert (published / 'second.zip').read_bytes() == b'second snapshot'


def test_publication_uses_frozen_plan_and_keeps_candidates_separate(tmp_path, results):
    purchase = _import_module('src.00_common.purchase_plan')
    candidates = pd.DataFrame([{
        'race_id': '202606040510', 'bet_type': 'win', 'selection': str(i), 'bet_type_label': '単勝',
        'estimated_probability': .5, 'odds_used': 3., 'recovery_rate_percent': 150.,
        'recommendation_score': float(i), 'bet_type_reliability': 1., 'expected_profit_per_100': 50.,
    } for i in range(1, 5)])
    frozen = purchase.select_plan(candidates)
    frozen['stake_yen'] = 200
    frozen.to_csv(results / 'recommended_bets.csv', index=False)
    candidates.loc[0, 'recommendation_score'] = 99999.
    candidates.to_csv(results / 'bets.csv', index=False)
    report_path = results / 'date_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['races'][0]['bet_rows'] = 4
    report['purchase_plan_version'] = 1
    report_path.write_text(json.dumps(report), encoding='utf-8')
    site = tmp_path / 'docs'
    prepare_publication(results, site)
    published = pd.read_csv(site / 'predictions/2026-09-19_bets.csv')
    assert published.selection.tolist() == [4, 3, 2]
    assert published.stake_yen.tolist() == [200, 200, 200]
    assert len(pd.read_csv(site / 'predictions/2026-09-19_candidates.csv')) == 4
    assert '<th>購入額</th>' in (site / 'predictions/2026-09-19.html').read_text(encoding='utf-8')


def test_missing_modern_purchase_plan_stops_publication(tmp_path, results):
    report_path = results / 'date_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['purchase_plan_version'] = 1
    report_path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='Missing saved purchase plan'):
        prepare_publication(results, tmp_path / 'docs')
    assert not (tmp_path / 'docs').exists()


# 前日照合が失敗しても当日予想の成功分は反映し、既存の前日結果は保持する。
@pytest.mark.parametrize("comparison_status", ["error", "incomplete"])
def test_successful_prediction_publishes_despite_failed_comparison(tmp_path, results, comparison_status):
    daily = _import_module('src.10_workflows.daily_racing')
    output = tmp_path / "run"
    output.mkdir()
    results.rename(output / "prediction")
    (output / "daily_report.json").write_text(json.dumps({
        "status": comparison_status, "publication_required": True,
        "tasks": {"prediction": {"status": "ok"}, "comparison": {"status": comparison_status}},
    }), encoding="utf-8")
    site = tmp_path / "docs"
    (site / "predictions").mkdir(parents=True)
    prior = site / "predictions/2026-09-18.html"
    prior.write_bytes(b"previous result must stay")
    assert daily.prepare_daily_publication(output, site)
    assert (site / "predictions/2026-09-19.csv").exists()
    assert (site / "predictions/2026-09-19.html").exists()
    assert prior.read_bytes() == b"previous result must stay"


# 取消後の有効な出走頭数を基準に公開検証し、取消前の頭数との違いで拒否しない。
def test_publication_uses_active_runner_count_after_withdrawal(tmp_path, results):
    date_workflow = _import_module('src.10_workflows.date_prediction')
    race_dir = tmp_path / "race"
    (race_dir / "fetch").mkdir(parents=True)
    (race_dir / "fetch/report.json").write_text(json.dumps({
        "card": {"status": "ok", "runners": 2}, "jra_odds": {"status": "ok"},
        "database": {"status": "ok", "expected_runners": 1, "withdrawn_horse_numbers": [6]},
    }), encoding="utf-8")
    path = results / "date_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["races"] = [date_workflow.race_row(report["race_ids"][0], {
        "status": "ok", "prediction_status": "ok", "odds_match_status": "ok",
        "betting_status": "ok", "prediction_rows": 1, "bet_rows": 0, "model_id": "m",
    }, race_dir)]
    path.write_text(json.dumps(report), encoding="utf-8")
    assert report["races"][0]["runners"] == 1
    assert prepare_publication(results, tmp_path / "docs")


# 不足や矛盾のある成果物を反映前に拒否し、既存公開ファイルの内容を保つ。
@pytest.mark.parametrize("problem", ["partial", "stage", "missing_race", "count", "nan_probability"])
def test_rejects_incomplete_or_inconsistent_results_without_touching_site(tmp_path, results, problem):
    path = results / "date_report.json"
    report = json.loads(path.read_text())
    if problem == "partial": report["status"] = "incomplete"
    if problem == "stage": report["races"][0]["database"] = "incomplete"
    if problem == "missing_race": report["race_ids"].append("202606040511")
    if problem == "count": report["races"][0]["runners"] = 2
    if problem == "nan_probability":
        data = pd.read_csv(results / "predictions.csv")
        data["top3_probability"] = float("nan")
        data.to_csv(results / "predictions.csv", index=False)
    path.write_text(json.dumps(report), encoding="utf-8")
    site = tmp_path / "docs"
    site.mkdir()
    (site / "index.html").write_bytes(b"keep")
    with pytest.raises(ValueError): prepare_publication(results, site)
    assert list(site.iterdir()) == [site / "index.html"]
    assert (site / "index.html").read_bytes() == b"keep"
