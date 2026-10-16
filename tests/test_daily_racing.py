# 検証対象：日本時間の日付、開催判定、当日予想と前日照合の独立した成功・失敗を確認する。
# 公開元の変更検出、取消馬・除外馬・中止馬の精算、不完全な結果の公開防止を検証する。
#
from importlib import import_module as _import_module
from calendar import monthrange
from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

daily = _import_module('src.10_workflows.daily_racing')
parse_meeting_day = _import_module('src.20_scrapers_html_collection.race_calendar').parse_meeting_day
build_prediction_site = _import_module('src.60_publication.static_site').build_prediction_site


# 指定年月の全日を含む開催カレンダーを作り、対象年月・日数の厳密な検査を確認する。
def calendar_html(year=2026, month=9):
    cells = []
    for day in range(1, monthrange(year, month)[1] + 1):
        body = f'<span class="Day">{day}</span>'
        if day == 19:
            body = f'<a href="race_list.html?kaisai_date={year}{month:02d}{day:02d}">{body}<span class="JyoName">中山</span></a>'
        cells.append(f'<td class="RaceCellBox">{body}</td>')
    return f'<select id="cal_select_year"><option value="{year}" selected>{year}</option></select><table class="Calendar_Table"><tr>'+''.join(cells)+'</tr></table>'


# 対象年月のカレンダーを使い、開催日と非開催日を区別できることを確認する。
def test_calendar_requires_complete_matching_month():
    html = calendar_html()
    assert parse_meeting_day(html, date(2026, 9, 19))
    assert not parse_meeting_day(html, date(2026, 9, 20))
    for invalid in ("<html>maintenance</html>", html.replace('>30</span>', '>29</span>'),
                    html.replace('20260919', '20260819'), html.replace('class="JyoName"', 'class="unknown"')):
        with pytest.raises(ValueError): parse_meeting_day(invalid, date(2026, 9, 20))


# UTCでは前日でも日本時間では翌日となる時刻と、年をまたぐ境界を確認する。
def test_japan_day_uses_local_date_and_year_boundary():
    assert daily.japan_today(datetime(2026, 12, 31, 18, tzinfo=timezone.utc)) == date(2027, 1, 1)
    assert daily.japan_today(datetime(2026, 9, 20, 18, tzinfo=timezone.utc)) == date(2026, 9, 21)


# カレンダーの取得・解析失敗を開催なしへ変換せず、前日タスクは指定どおり独立して実行する。
@pytest.mark.parametrize("today,previous,expected,publication", [
    (False, False, "ok", False),
    (False, True, "ok", True),
    (True, False, "ok", True),
    ("error", True, "error", True),
    (True, "error", "error", True),
    ("incomplete", True, "incomplete", True),
    ("error", "error", "error", False),
    ("incomplete", False, "incomplete", False),
])
def test_daily_calendar_errors_never_become_nonmeeting(tmp_path, monkeypatch, today, previous, expected, publication):
    called = []
    # 開催日の判定結果を固定し、ワークフローの分岐を検証する。
    def meeting(day, *args):
        status = today if day.day == 21 else previous
        if status == "error": raise ValueError("calendar unavailable")
        return bool(status)
    monkeypatch.setattr(daily, "check_meeting_day", meeting)
    monkeypatch.setattr(daily, "run_date", lambda *a: {"status": "incomplete" if today == "incomplete" else "ok"})
    # 結果照合の代替処理。呼出しや返す状態をテスト側で制御する。
    def compare(*args):
        called.append(args[0])
        return {"status": "ok"}
    monkeypatch.setattr(daily, "compare_previous", compare)
    output = tmp_path / "run"
    report = daily.run_daily(date(2026, 9, 21), tmp_path, tmp_path, output, True)
    assert report["status"] == expected
    assert report["publication_required"] is publication
    assert bool(called) == (previous is True)
    if not publication:
        if expected == "ok":
            assert daily.prepare_daily_publication(output, tmp_path / "missing_site") is False
        else:
            with pytest.raises(ValueError): daily.prepare_daily_publication(output, tmp_path / "missing_site")


# 前日の予想と買い目を持つ一時サイトを準備し、結果取得も固定応答へ差し替えるfixture。
@pytest.fixture
def prior_site(tmp_path, monkeypatch):
    site = tmp_path / "docs"
    (site / "predictions").mkdir(parents=True)
    row = {"race_id": "202606040510", "race_date": "2026-09-19", "horse_id": "h1", "horse_number": 1,
           "model_run_id": "m", "top3_probability": 0.5, "course_name": "中山", "race_number": 10,
           "race_name": "Test", "horse_name": "Horse", "jockey_name": "Jockey", "prediction_rank": 1,
           "odds": 3.0, "expected_value_index": 1.5}
    prediction = pd.DataFrame([row])
    prediction.to_csv(site / "predictions/2026-09-19.csv", index=False)
    pd.DataFrame([{"race_id": row["race_id"], "bet_type": "win", "selection": "1",
                   "bet_type_label": "単勝", "estimated_probability": 0.5, "odds_used": 3.0,
                   "recovery_rate_percent": 150.0, "recommendation_score": 150.0,
                   "expected_profit_per_100": 50.0,
                   "is_primary_bet_type": False, "bet_type_reliability": 1.0}]).to_csv(site / "predictions/2026-09-19_bets.csv", index=False)
    build_prediction_site(prediction, "2026-09-19", "m", site)
    monkeypatch.setattr(daily.DateCollector, "race_ids_for_date", lambda *a, **kw: [row["race_id"]])
    # 外部取得を代替し、テストで指定した公式結果または取得失敗を返す。
    def fetch(*args, **kwargs):
        actual = pd.DataFrame([{"horse_id": None, "horse_number": 1, "finish_position": 1}])
        actual.attrs["official_payouts"] = {"win:1": 300}
        return actual
    monkeypatch.setattr(daily.JraResultFetcher, "fetch_result_for_comparison", fetch)
    return site


def test_legacy_candidates_settle_only_displayed_tickets(tmp_path, prior_site):
    import logging
    source = prior_site / 'predictions/2026-09-19_bets.csv'
    template = pd.read_csv(source).iloc[0].to_dict()
    candidates = pd.DataFrame([{**template, 'selection': str(i), 'recommendation_score': 500 - index}
                               for index, i in enumerate([2, 3, 4, 1])])
    candidates.to_csv(source, index=False)
    # 唯一の的中候補1は4位。HTMLに載らない候補を精算・回収率へ混ぜない。
    result = daily.compare_previous(date(2026, 9, 19), prior_site, tmp_path / 'comparison', logging.getLogger())
    judged = pd.read_csv(tmp_path / 'comparison/bets_results.csv')
    assert judged.selection.tolist() == [2, 3, 4]
    assert result['summary']['bet_returns']['purchase'] == 300
    assert result['summary']['bet_returns']['payout'] == 0
    assert result['summary']['bet_returns']['recovery_rate'] == 0


def test_frozen_primary_stake_is_used_by_daily_settlement(tmp_path, prior_site):
    import logging
    purchase = _import_module('src.00_common.purchase_plan')
    source = prior_site / 'predictions/2026-09-19_bets.csv'
    frozen = purchase.select_plan(pd.read_csv(source, dtype={'selection': str, 'race_id': str}))
    frozen['stake_yen'] = 300
    frozen.to_csv(source, index=False)
    result = daily.compare_previous(date(2026, 9, 19), prior_site, tmp_path / 'comparison', logging.getLogger())
    assert result['summary']['bet_returns']['purchase'] == 300
    assert result['summary']['bet_returns']['payout'] == 900
    assert result['summary']['bet_returns']['recovery_rate'] == 300


def test_changed_resolved_plan_blocks_result_publication(tmp_path, prior_site):
    import logging
    output = tmp_path / 'run'
    output.mkdir()
    daily.compare_previous(date(2026, 9, 19), prior_site, output / 'comparison', logging.getLogger())
    (output / 'daily_report.json').write_text(json.dumps({
        'status': 'ok', 'tasks': {'prediction': {'status': 'no_races'},
        'comparison': {'status': 'ok', 'race_date': '2026-09-19'}}}), encoding='utf-8')
    source = output / 'comparison/purchase_plan.csv'
    source.write_bytes(source.read_bytes() + b'\n')
    before = {str(p): p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='purchase plan changed'):
        daily.prepare_daily_publication(output, prior_site)
    assert before == {str(p): p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}


# 公開済み予想を再計算せずに照合し、元CSVを保ったまま比較結果だけを追加できることを確認する。
@pytest.mark.parametrize("prediction_status", ["no_races", "incomplete", "error"])
def test_previous_results_update_without_repredicting_and_preserve_original(tmp_path, prior_site, prediction_status):
    import logging
    output = tmp_path / "run"
    output.mkdir()
    original = (prior_site / "predictions/2026-09-19.csv").read_bytes()
    result = daily.compare_previous(date(2026, 9, 19), prior_site, output / "comparison", logging.getLogger())
    assert result["status"] == "ok"
    (output / "daily_report.json").write_text(json.dumps({
        "status": "ok" if prediction_status == "no_races" else prediction_status,
        "publication_required": prediction_status == "no_races",
        "tasks": {"prediction": {"status": prediction_status}, "comparison": {"status": "ok", "race_date": "2026-09-19"}}}), encoding="utf-8")
    assert daily.prepare_daily_publication(output, prior_site)
    assert (prior_site / "predictions/2026-09-19.csv").read_bytes() == original
    assert (prior_site / "predictions/2026-09-19_comparison.csv").exists()
    payouts = pd.read_csv(prior_site / "predictions/2026-09-19_bets_results.csv")
    assert payouts.payout_per_100.tolist() == [300]
    snapshot = {p.name:p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}
    daily.prepare_daily_publication(output, prior_site)
    assert snapshot == {p.name:p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}


# 結果の着順が不足したレースは照合完了とせず、公開可能な成功へ数えない。
def test_result_missing_finish_stops_publication(tmp_path, monkeypatch, prior_site):
    import logging
    monkeypatch.setattr(daily.JraResultFetcher, "fetch_result_for_comparison", lambda *a, **kw:
                        pd.DataFrame([{"horse_id": None, "horse_number": 1, "finish_position": None}]))
    result = daily.compare_previous(date(2026, 9, 19), prior_site, tmp_path / "comparison", logging.getLogger())
    assert result["status"] == "incomplete"
    assert result["summary"]["failed_races"] == 1


# 前日予想のファイルがない場合に、不完全な精算を開始せず明示的に失敗することを確認する。
def test_missing_prior_prediction_is_error(tmp_path):
    import logging
    with pytest.raises(ValueError, match="CSV"):
        daily.compare_previous(date(2026, 9, 19), tmp_path, tmp_path / "comparison", logging.getLogger())


def test_calibration_returns_reuse_results_and_reach_public_html(tmp_path, monkeypatch, prior_site):
    import logging
    day = '2026-09-19'
    predictions = pd.read_csv(prior_site / f'predictions/{day}.csv', dtype={'race_id': str, 'horse_id': str})
    bets = pd.read_csv(prior_site / f'predictions/{day}_bets.csv', dtype={'race_id': str, 'selection': str})
    root = prior_site / f'predictions/calibration/{day}'
    root.mkdir(parents=True)
    (root / 'status.json').write_text(json.dumps({m: {'status': 'ok'} for m in ('class_weight', 'sigmoid')}))
    for method in ('class_weight', 'sigmoid'):
        target = root / method
        target.mkdir()
        predictions.assign(calibration_method=method, calibration_id='cal_' + method).to_csv(target / 'predictions.csv', index=False)
        shadow_bets = bets.assign(calibration_method=method, calibration_id='cal_' + method,
                                  bet_type='win' if method == 'class_weight' else 'place')
        shadow_bets.to_csv(target / 'bets.csv', index=False)
        shadow_bets.to_csv(target / 'recommended_bets.csv', index=False)
    calls = []
    def fetch(self, race_id, *args, **kwargs):
        calls.append(race_id)
        frame = pd.DataFrame([{'horse_id': None, 'horse_number': 1, 'finish_position': 1}])
        frame.attrs['official_payouts'] = {'win:1': 300, 'place:1': 150}
        return frame
    monkeypatch.setattr(daily.JraResultFetcher, 'fetch_result_for_comparison', fetch)
    output = tmp_path / 'run'
    result = daily.compare_previous(date.fromisoformat(day), prior_site, output / 'comparison', logging.getLogger())
    assert calls == [predictions.race_id.iloc[0]]  # 比較方式のための追加通信はしない。
    assert result['calibration_returns']['class_weight']['recovery_rate'] == 300
    assert result['calibration_returns']['sigmoid']['recovery_rate'] == 150
    (output / 'daily_report.json').write_text(json.dumps({'status': 'ok', 'tasks': {
        'prediction': {'status': 'no_races'}, 'comparison': {'status': 'ok', 'race_date': day}}}))
    assert daily.prepare_daily_publication(output, prior_site)
    html = (prior_site / f'predictions/{day}.html').read_text(encoding='utf-8')
    assert '重み付け補正【検証中】' in html and 'sigmoid補正【検証中】' in html
    assert '回収率 300.0%' in html and '回収率 150.0%' in html
    assert (root / 'returns.json').is_file()
    assert (root / 'sigmoid/bets_results.csv').is_file()
    assert pd.read_csv(root / 'sigmoid/recommended_bets.csv').bet_type.iloc[0] == 'place'
    # 公開後に入力が変わった場合は、同じ比較結果で再公開せず停止する。
    with (root / 'sigmoid/recommended_bets.csv').open('a') as stream:
        stream.write('\n')
    before = (prior_site / f'predictions/{day}.html').read_bytes()
    with pytest.raises(ValueError, match='変更'):
        daily.prepare_daily_publication(output, prior_site)
    assert (prior_site / f'predictions/{day}.html').read_bytes() == before


# 着順があっても公式払戻金が不足していれば、照合を完了扱いにしない。
def test_missing_payout_stops_comparison(tmp_path, monkeypatch, prior_site):
    import logging
    monkeypatch.setattr(daily.JraResultFetcher, "fetch_result_for_comparison", lambda *a, **kw:
                        pd.DataFrame([{"horse_id": None, "horse_number": 1, "finish_position": 1}]))
    result = daily.compare_previous(date(2026, 9, 19), prior_site, tmp_path / "comparison", logging.getLogger())
    assert result["status"] == "incomplete"


# 照合開始後に元CSVが変わった場合、予想と照合の両方について既存サイトを変更しない。
def test_changed_previous_inputs_prevent_both_publications(tmp_path, monkeypatch, prior_site):
    import logging
    output = tmp_path / "run"
    output.mkdir()
    daily.compare_previous(date(2026, 9, 19), prior_site, output / "comparison", logging.getLogger())
    (output / "daily_report.json").write_text(json.dumps({"status": "ok", "publication_required": True,
        "tasks": {"prediction": {"status": "ok"}, "comparison": {"status": "ok", "race_date": "2026-09-19"}}}), encoding="utf-8")
    # 当日予想の代替成果物を用意し、公開準備の成否を検証する。
    def prepare_today(source, staged):
        (staged / "predictions/today.html").write_bytes(b"new prediction")
    monkeypatch.setattr(daily, "prepare_publication", prepare_today)
    with (prior_site / "predictions/2026-09-19.csv").open("ab") as stream:
        stream.write(b"\n")
    before = {p.name:p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match="変更"):
        daily.prepare_daily_publication(output, prior_site)
    assert before == {p.name:p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}


# 実例HTMLの除外・中止を保持して日次精算を完了し、公開ページにも状態を表示する。
def test_real_excluded_and_stopped_races_complete_daily_run_and_publication(tmp_path, monkeypatch):
    """Replay the two failed races using their published predictions and JRA HTML."""
    repo = Path(__file__).resolve().parents[1]
    ids = {"202609040704", "202609040706"}
    site = tmp_path / "docs"
    (site / "predictions").mkdir(parents=True)
    for suffix in ("", "_bets"):
        name = f"2026-09-21{suffix}.csv"
        frame = pd.read_csv(repo / "docs/predictions" / name,
                            dtype={"race_id": str, "horse_id": str, "selection": str})
        frame[frame.race_id.isin(ids)].to_csv(site / "predictions" / name, index=False)
    parser = _import_module('src.20_scrapers_html_collection.jra_results').parse_jra_result_html
    # 外部取得を代替し、テストで指定した公式結果または取得失敗を返す。
    def fetch(self, race_id, *args, **kwargs):
        return parser(repo / "tests/fixtures/jra" / f"{race_id}.html", race_id)
    monkeypatch.setattr(daily.JraResultFetcher, "fetch_result_for_comparison", fetch)
    monkeypatch.setattr(daily.DateCollector, "race_ids_for_date", lambda *a, **kw: sorted(ids))
    monkeypatch.setattr(daily, "check_meeting_day", lambda day, *a: day == date(2026, 9, 21))
    output = tmp_path / "run"
    result = daily.run_daily(date(2026, 9, 22), tmp_path, site, output, True)
    assert result["status"] == "ok"
    assert result["publication_required"]
    report = json.loads((output / "comparison/report.json").read_text(encoding="utf-8"))
    assert report["summary"]["compared_races"] == 2
    assert report["summary"]["failed_races"] == 0
    comparison = pd.read_csv(output / "comparison/comparison.csv", dtype={"race_id": str})
    assert set(comparison.result_status) == {"finished", "excluded", "did_not_finish"}
    # 推奨買い目へ的中・不的中・返還・未確認と公式払戻額を付けた表。
    judged = pd.read_csv(output / "comparison/bets_results.csv")
    assert not judged.bet_result.eq("未確認").any()
    assert daily.prepare_daily_publication(output, site)
    html = (site / "predictions/2026-09-21.html").read_text(encoding="utf-8")
    assert "<td>除外</td>" in html
    assert "<td>中止</td>" in html
    assert 'data-comparison-status="completed"' in html


# 予想前の取消と予想後の取消を分け、公式返還情報が必要な買い目は返還まで確認する。
@pytest.mark.parametrize("withdrawn_before_prediction", [False, True])
@pytest.mark.parametrize("refund_listed", [False, True])
def test_daily_settles_withdrawals_before_and_after_prediction(tmp_path, monkeypatch, prior_site, withdrawn_before_prediction, refund_listed):
    import logging
    race_id = "202606040510"
    predictions_path = prior_site / "predictions/2026-09-19.csv"
    if not withdrawn_before_prediction:
        predictions = pd.read_csv(predictions_path, dtype={"race_id": str})
        withdrawn = predictions.iloc[0].copy()
        withdrawn["horse_number"] = 6
        withdrawn["horse_id"] = "h6"
        withdrawn["prediction_rank"] = 2
        pd.concat([predictions, withdrawn.to_frame().T]).to_csv(predictions_path, index=False)
        bets_path = prior_site / "predictions/2026-09-19_bets.csv"
        bets = pd.read_csv(bets_path, dtype={"race_id": str, "selection": str})
        bets.loc[0, "selection"] = "6"
        bets.to_csv(bets_path, index=False)
    # 外部取得を代替し、テストで指定した公式結果または取得失敗を返す。
    def fetch(*args, **kwargs):
        actual = pd.DataFrame([
            {"horse_id": None, "horse_number": 1, "finish_position": 1, "result_status": "finished"},
            {"horse_id": None, "horse_number": 6, "finish_position": None, "result_status": "scratched"},
        ])
        actual.attrs["official_payouts"] = {"win:1": 300}
        actual.attrs["official_refunds"] = {"horse_numbers": [6] if refund_listed else []}
        return actual
    monkeypatch.setattr(daily.JraResultFetcher, "fetch_result_for_comparison", fetch)
    output = tmp_path / "comparison"
    report = daily.compare_previous(date(2026, 9, 19), prior_site, output, logging.getLogger(__name__))
    if not withdrawn_before_prediction and not refund_listed:
        assert report["status"] == "incomplete"
        assert report["summary"]["failed_races"] == 1
        return
    assert report["status"] == "ok"
    # 推奨買い目へ的中・不的中・返還・未確認と公式払戻額を付けた表。
    judged = pd.read_csv(output / "bets_results.csv")
    assert judged.iloc[0].bet_result == ("的中" if withdrawn_before_prediction else "返還")
    assert judged.iloc[0].payout_per_100 == (300 if withdrawn_before_prediction else 100)


# 取消ではない出走馬が予想に欠けていれば、対象馬不一致として照合を止める。
def test_unpredicted_active_horse_still_blocks_comparison(tmp_path, monkeypatch, prior_site):
    import logging
    # 外部取得を代替し、テストで指定した公式結果または取得失敗を返す。
    def fetch(*args, **kwargs):
        actual = pd.DataFrame([
            {"horse_id": None, "horse_number": 1, "finish_position": 1},
            {"horse_id": None, "horse_number": 6, "finish_position": 2},
        ])
        actual.attrs["official_payouts"] = {"win:1": 300}
        return actual
    monkeypatch.setattr(daily.JraResultFetcher, "fetch_result_for_comparison", fetch)
    report = daily.compare_previous(date(2026, 9, 19), prior_site, tmp_path / "comparison", logging.getLogger(__name__))
    assert report["status"] == "incomplete"
    assert report["summary"]["failed_races"] == 1
    assert "馬番が一致" in report["summary"]["failures"][0]["message"]
