from datetime import date
from importlib import import_module

import pytest


JraOddsHtmlCollector = import_module(
    "src.20_scrapers_html_collection.jra_odds_html"
).JraOddsHtmlCollector


class LoggerStub:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def test_extracts_jra_race_id_and_bet_type():
    parsed = JraOddsHtmlCollector._race_id_and_bet_type(
        "pw154ouS306202604050120260919Z/8B",
        date(2026, 9, 19),
    )
    assert parsed == ("202606040501", "quinella")


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
