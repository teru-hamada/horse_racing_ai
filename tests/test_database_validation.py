# 検証対象：検証用DBへの登録・読み戻し・再登録の一致と、本番DBからの隔離を確認する。
# 取消馬の除外、単勝との結合、重複や不足の検出を異常ケースも含めて検証する。
#
from importlib import import_module as _import_module
from datetime import date
from dataclasses import replace

import pandas as pd
import pytest

smoke = _import_module('src.10_workflows.database_validation')


# テストが共有する出馬表やオッズなどの小さな入力表を返すfixture。
@pytest.fixture
def frames():
    common = {"race_id": "202609040701", "race_date": date(2026, 9, 21)}
    card = pd.DataFrame([
        {**common, "horse_id": str(100 + n), "horse_name": f"Horse {n}",
         "horse_number": n, "frame_number": n} for n in (1, 2)
    ])
    odds = pd.DataFrame([
        {**common, "bet_type": kind, "selection_1": n, "selection_2": None,
         "selection_3": None, "odds_min": 2.5, "odds_max": 3.0}
        for n in (1, 2) for kind in ("win", "place")
    ])
    return card, odds


# 検証処理の代替。正常・例外を切り替え、CLIの終了状態を確認する。
def verify(frames, output):
    return smoke.verify_database(*frames, date(2026, 9, 21), "202609040701", output)


# 一時DBへ保存・読み戻し・再登録を行い、値の一致と本番DBのバイト列不変を確認する。
def test_real_storage_round_trip_is_isolated_and_idempotent(tmp_path, monkeypatch, frames):
    production = tmp_path / "production.duckdb"
    production.write_bytes(b"must not open or modify")
    monkeypatch.setattr(smoke.storage, "PATHS", replace(smoke.storage.PATHS, database=production))
    report = verify(frames, tmp_path)
    assert report["status"] == "ok"
    assert all(report["checks"].values())
    assert report["matched_win_runners"] == 2
    assert report["registered_odds_by_bet_type"] == {"place": 2, "win": 2}
    assert production.read_bytes() == b"must not open or modify"
    assert (tmp_path / "smoke.duckdb").exists()
    assert (tmp_path / "db_win_join.csv").exists()
    with pytest.raises(ValueError):
        verify(frames, tmp_path)


# 保存済みデータに不足や不正を与え、該当する検査名とincomplete状態が報告されることを確認する。
@pytest.mark.parametrize("problem,failed_check", [
    ("missing_numbers", "registration_1_horse_and_frame_numbers"),
    ("duplicate_numbers", "registration_1_horse_and_frame_numbers"),
    ("missing_win", "win_odds_match_all_runners"),
    ("wrong_horse", "win_odds_match_all_runners"),
    ("wrong_race", "registration_1_odds_counts"),
    ("wrong_date", "registration_1_race_and_date"),
    ("duplicate_odds", "unique_odds_selections"),
    ("empty_odds", "registration_1_odds_counts"),
])
def test_detects_unusable_registered_data(tmp_path, frames, problem, failed_check):
    card, odds = frames
    if problem == "missing_numbers":
        card["horse_number"] = None
    elif problem == "duplicate_numbers":
        card["horse_number"] = 1
    elif problem == "missing_win":
        odds = odds[~(odds.bet_type.eq("win") & odds.selection_1.eq(2))]
    elif problem == "wrong_horse":
        odds.loc[odds.selection_1.eq(2), "selection_1"] = 3
    elif problem == "wrong_race":
        odds["race_id"] = "202609040702"
    elif problem == "wrong_date":
        odds["race_date"] = date(2026, 9, 20)
    elif problem == "duplicate_odds":
        odds = pd.concat([odds, odds.iloc[:1]], ignore_index=True)
    elif problem == "empty_odds":
        odds = odds.iloc[:0]
    report = verify((card, odds), tmp_path)
    assert report["status"] == "incomplete"
    assert failed_check in report["failed_checks"]


# 再登録で置換せず重複追記する実装に差し替え、冪等性の検査が異常を検出することを確認する。
def test_detects_duplicate_append_on_reregistration(tmp_path, monkeypatch, frames):
    original = smoke.storage.save_race_odds
    # 保存後に重複行を意図的に追加し、再登録の整合性検査が異常を見抜けるか調べる。
    def append_after_save(frame, run_id, **kwargs):
        original(frame, run_id, **kwargs)
        if run_id.endswith("2"):
            with smoke.storage.connect(kwargs["database"]) as con:
                con.execute("INSERT INTO race_odds SELECT * FROM race_odds")
    monkeypatch.setattr(smoke.storage, "save_race_odds", append_after_save)
    report = verify(frames, tmp_path)
    assert report["status"] == "incomplete"
    assert "reregistration_unchanged" in report["failed_checks"]


# 取消馬の出馬表とオッズを予想用DBから外し、残った出走馬だけが対応することを確認する。
@pytest.mark.parametrize("status", ["scratched", "excluded"])
@pytest.mark.parametrize("stale_odds", [False, True])
def test_withdrawn_horses_are_not_in_prediction_database(tmp_path, frames, status, stale_odds):
    card, odds = frames
    card["entry_status"] = ["active", status]
    if not stale_odds:
        odds = odds[odds.selection_1.eq(1)]
    report = verify((card, odds), tmp_path)
    assert report["status"] == "ok"
    assert report["withdrawn_horse_numbers"] == [2]
    assert report["matched_win_runners"] == 1
    assert report["registered_runners"] == 1
    loaded = smoke.storage.load_records("upcoming", database=tmp_path / "smoke.duckdb")
    assert loaded.horse_number.tolist() == [1]
    # 検証用DBから読み戻したオッズ。券種数・買い目キー・入力値を照合する。
    loaded_odds = smoke.storage.load_race_odds([card.race_id.iloc[0]], database=tmp_path / "smoke.duckdb")
    assert set(loaded_odds.selection_1) == {1}


# 16頭のうち6番が取消の場合も、馬番の欠番を許容して15頭の出馬表とオッズを照合する。
def test_sixteen_entries_with_number_six_scratched(tmp_path, frames):
    template_card, template_odds = frames
    card = pd.DataFrame([{**template_card.iloc[0].to_dict(), "horse_id": str(n),
                          "horse_number": n, "frame_number": (n + 1) // 2,
                          "entry_status": "scratched" if n == 6 else "active"}
                         for n in range(1, 17)])
    odds = pd.DataFrame([{**template_odds.iloc[0].to_dict(), "selection_1": n}
                         for n in range(1, 17) if n != 6])
    report = verify((card, odds), tmp_path)
    assert report["status"] == "ok"
    assert report["registered_runners"] == report["matched_win_runners"] == 15
    assert report["withdrawn_horse_numbers"] == [6]
    assert report["unmatched_card_numbers"] == []


# DB診断の成否が終了コード、JSON、サマリーへ一貫して反映されることを確認する。
@pytest.mark.parametrize("database_error", [False, True])
def test_cli_database_mode_reports_success_and_errors(tmp_path, monkeypatch, frames, database_error):
    import json
    html = _import_module('src.10_workflows.html_acquisition')

    card, odds = frames
    monkeypatch.setattr(html.NetkeibaHtmlCollector, "_download", lambda *a, **kw: "<html></html>")
    monkeypatch.setattr(html.NetkeibaHtmlCollector, "_enrich_upcoming_html_with_odds", lambda self, text, *a, **kw: text)
    monkeypatch.setattr(html.NetkeibaHtmlCollector, "_parse_page", lambda *a, **kw: card)
    monkeypatch.setattr(html.DiagnosticOddsCollector, "collect_date", lambda *a, **kw: 1)
    monkeypatch.setattr(html, "parse_odds_directory", lambda *a: odds)
    if database_error:
        # 対象の失敗経路へ進ませるため、呼び出されたら意図的に例外を発生させる。
        def fail(*a, **kw):
            raise RuntimeError("simulated database failure")
        monkeypatch.setattr(smoke.storage, "save_race_records", fail)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    output = tmp_path / "output"
    code = html.main(["--race-date", "2026-09-21", "--race-id", "202609040701",
                      "--output", str(output), "--verify-db"])
    assert code == (1 if database_error else 0)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["database"]["status"] == ("error" if database_error else "ok")
    assert "database" in summary.read_text(encoding="utf-8")
