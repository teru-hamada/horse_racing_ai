# 検証対象：JRA画面の遷移識別子から対象レースと券種を選び、HTMLを保存する流れを確認する。
# 通信を偽の応答に置き換え、UTF-8宣言、進捗通知、開催なしの場合の挙動を検証する。
#
from datetime import date
from importlib import import_module

import pytest


JraOddsHtmlCollector = import_module(
    "src.20_scrapers_html_collection.jra_odds_html"
).JraOddsHtmlCollector


# テスト中のログを記録する代替ロガー。実際のログ出力先に依存せず通知内容を検証する。
class LoggerStub:
    # テスト用の入力・応答・呼出履歴を初期化し、ケースごとに独立した状態を用意する。
    def __init__(self):
        self.messages = []

    # 通常の処理経過をINFOレベルでファイルと任意の通知先へ送る。
    def info(self, message):
        self.messages.append(("info", message))

    # 処理継続は可能だが確認が必要な状態をWARNINGとして通知する。
    def warning(self, message):
        self.messages.append(("warning", message))


# JRA画面遷移用の識別子から、レースIDと券種を読み分ける。
def test_extracts_jra_race_id_and_bet_type():
    parsed = JraOddsHtmlCollector._race_id_and_bet_type(
        "pw154ouS306202604050120260919Z/8B",
        date(2026, 9, 19),
    )
    assert parsed == ("202606040501", "quinella")


# 対象日の対象レースだけを保存し、保存HTMLの文字コード宣言をUTF-8へそろえる。
def test_collect_date_saves_only_target_race_odds(tmp_path):
    collector = JraOddsHtmlCollector(LoggerStub(), interval_seconds=0)
    responses = {
        collector.ENTRY_CNAME: """
          <a onclick="return doAction('/JRADB/accessO.html',
             'pw15orl00062026040520260919/A1');">中山</a>
        """,
        "pw15orl00062026040520260919/A1": """
          <a onclick="return doAction('/JRADB/accessO.html',
             'pw151ouS306202604050120260919Z/FF');">単複</a>
          <a onclick="return doAction('/JRADB/accessO.html',
             'pw154ouS306202604050120260919Z/8B');">馬連</a>
          <a onclick="return doAction('/JRADB/accessO.html',
             'pw154ouS306202604050220260919Z/40');">別レース</a>
        """,
        "pw151ouS306202604050120260919Z/FF": (
            '<html><meta charset="Shift_JIS"><table class="tanpuku"></table></html>'
        ),
        "pw154ouS306202604050120260919Z/8B": (
            '<html><meta charset="Shift_JIS"><table class="umaren"></table></html>'
        ),
    }
    collector._post = responses.__getitem__

    count = collector.collect_date(
        date(2026, 9, 19), ["202606040501"], tmp_path
    )

    assert count == 2
    win_place = tmp_path / "202606040501_win_place.html"
    quinella = tmp_path / "202606040501_quinella.html"
    assert win_place.exists()
    assert quinella.exists()
    assert 'charset="UTF-8"' in win_place.read_text(encoding="utf-8")
    assert "umaren" in quinella.read_text(encoding="utf-8")


def test_http_fetch_time_and_hash_survive_cache_reuse(tmp_path):
    import json
    from hashlib import sha256
    import pandas as pd

    collector = JraOddsHtmlCollector(LoggerStub(), interval_seconds=0)
    entry = "doAction('/JRADB/accessO.html', 'pw15orl00062026040520260919/A1')"
    venue = "doAction('/JRADB/accessO.html', 'pw151ouS306202604050120260919Z/FF')"

    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    pages = {collector.ENTRY_CNAME: entry, 'pw15orl00062026040520260919/A1': venue,
             'pw151ouS306202604050120260919Z/FF': '<html>odds</html>'}
    requests = []

    def post(url, *, data, **kwargs):
        requests.append(data['cname'])
        return Response(pages[data['cname']])

    collector.session.post = post
    assert collector.collect_date(date(2026, 9, 19), ['202606040501'], tmp_path) == 1
    path = tmp_path / '202606040501_win_place.html'
    sidecar = path.with_suffix('.metadata.json')
    saved = sidecar.read_bytes()
    metadata = json.loads(saved)
    assert pd.Timestamp(metadata['fetched_at']).tzinfo is not None
    assert metadata['source_sha256'] == sha256(path.read_bytes()).hexdigest()
    assert metadata['source_url'] == collector.URL
    requests.clear()
    assert collector.collect_date(date(2026, 9, 19), ['202606040501'], tmp_path) == 1
    assert sidecar.read_bytes() == saved
    assert 'pw151ouS306202604050120260919Z/FF' not in requests


# 探索と取得の各段階で、呼出側へ想定した進捗率が通知されることを確認する。
def test_collect_date_reports_odds_progress(tmp_path):
    collector = JraOddsHtmlCollector(LoggerStub(), interval_seconds=0)
    responses = {
        collector.ENTRY_CNAME: (
            "doAction('/JRADB/accessO.html', "
            "'pw15orl00062026040520260919/A1')"
        ),
        "pw15orl00062026040520260919/A1": (
            "doAction('/JRADB/accessO.html', "
            "'pw151ouS306202604050120260919Z/FF')"
        ),
        "pw151ouS306202604050120260919Z/FF": "<html>odds</html>",
    }
    collector._post = responses.__getitem__
    progress = []

    collector.collect_date(
        date(2026, 9, 19),
        ["202606040501"],
        tmp_path,
        progress_callback=progress.append,
    )

    assert progress == pytest.approx([0.1, 0.3, 1.0])


# 開催画面がなければ警告とゼロ件を返し、架空のHTMLファイルを作らない。
def test_collect_date_returns_zero_when_no_meeting_is_available(tmp_path):
    logger = LoggerStub()
    collector = JraOddsHtmlCollector(logger, interval_seconds=0)
    collector._post = lambda cname: "<html>発売前</html>"

    count = collector.collect_date(
        date(2026, 9, 20), ["202606040601"], tmp_path
    )

    assert count == 0
    assert not list(tmp_path.glob("*.html"))
    assert any(level == "warning" for level, _ in logger.messages)
