from datetime import date

from src.public_api import (
    AppLogger,
    NetkeibaDatabaseCreator,
    NetkeibaHtmlCollector,
)
from importlib import import_module


_passing_positions = import_module(
    "src.00_common.netkeiba_common"
)._passing_positions


RESULT_HTML = """
<html><head><title>2026年7月12日 福島11R</title></head><body>
<div class="RaceName">七夕賞</div>
<div class="RaceData01">芝2000m / 天候 : 晴 / 芝 : 良</div>
<div class="RaceData02">2回 福島 6日目</div>
<table class="RaceTable01">
<thead><tr><th>着順</th><th>枠</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th><th>騎手</th><th>タイム</th><th>通過</th><th>単勝</th><th>人気</th><th>馬体重</th><th>厩舎</th></tr></thead>
<tbody>
<tr class="HorseList"><td>1</td><td>3</td><td>5</td><td><a href="/horse/2020100001">テストホース</a></td><td>牡4</td><td>57.0</td><td><a href="/jockey/01234">テスト騎手</a></td><td>1:59.8</td><td>4-3-2-1</td><td>3.2</td><td>1</td><td>480(+4)</td><td><a href="/trainer/01001">テスト厩舎</a></td></tr>
</tbody></table>
</body></html>
"""

PEDIGREE_HTML = """
<html><body>
<table class="blood_table">
<tr>
  <td rowspan="4"><a href="/horse/ped/sire001/">父馬</a></td>
  <td rowspan="2"><a href="/horse/ped/ss001/">父父</a></td>
  <td><a href="/horse/ped/sss001/">父父父</a></td>
</tr>
<tr><td><a href="/horse/ped/ssd001/">父父母</a></td></tr>
<tr>
  <td rowspan="2"><a href="/horse/ped/sd001/">父母</a></td>
  <td><a href="/horse/ped/sds001/">父母父</a></td>
</tr>
<tr><td><a href="/horse/ped/sdd001/">父母母</a></td></tr>
<tr>
  <td rowspan="4"><a href="/horse/ped/dam001/">母馬</a></td>
  <td rowspan="2"><a href="/horse/ped/damsire001/">母父馬</a></td>
  <td><a href="/horse/ped/dss001/">母父父</a></td>
</tr>
<tr><td><a href="/horse/ped/dsd001/">母父母</a></td></tr>
<tr>
  <td rowspan="2"><a href="/horse/ped/dd001/">母母</a></td>
  <td><a href="/horse/ped/dds001/">母母父</a></td>
</tr>
<tr><td><a href="/horse/ped/ddd001/">母母母</a></td></tr>
</table>
</body></html>
"""


def test_result_parser(tmp_path):
    logger = AppLogger(tmp_path)
    scraper = NetkeibaDatabaseCreator(logger, interval_seconds=0.5)
    parsed = scraper._parse_page(RESULT_HTML, "202603020611", date(2026, 7, 12), "historical")
    assert len(parsed) == 1
    row = parsed.iloc[0]
    assert row["horse_id"] == "2020100001"
    assert row["horse_name"] == "テストホース"
    assert row["finish_position"] == 1
    assert row["surface"] == "芝"
    assert row["distance"] == 2000
    assert row["time_seconds"] == 119.8
    assert row["passing_position_1"] == 4
    assert row["passing_position_2"] == 3
    assert row["passing_position_3"] == 2
    assert row["passing_position_4"] == 1


def test_passing_positions_pads_missing_corners():
    assert _passing_positions("7-5") == (7, 5, None, None)
    assert _passing_positions(None) == (None, None, None, None)


def test_result_parser_ignores_broken_auxiliary_table(tmp_path):
    html = RESULT_HTML.replace(
        "</body>",
        "<table class=\"pay_table_01\"><thead></thead></table></body>",
    )
    logger = AppLogger(tmp_path)
    scraper = NetkeibaDatabaseCreator(logger, interval_seconds=0.5)

    parsed = scraper._parse_page(
        html,
        "202603020611",
        date(2026, 7, 12),
        "historical",
    )

    assert len(parsed) == 1
    assert parsed.iloc[0]["horse_id"] == "2020100001"


def test_upcoming_odds_are_embedded_in_saved_html():
    html = """
    <html><body>
      <span id="odds-1_01">---.-</span>
      <span id="ninki-1_01">**</span>
      <span id="odds-1_02">---.-</span>
      <span id="ninki-1_02">**</span>
    </body></html>
    """
    enriched, count = NetkeibaHtmlCollector._embed_win_odds(
        html,
        {
            "01": ["3.2", "0", "1"],
            "02": ["8.6", "0", "2"],
        },
        "2026-07-26 09:30:00",
    )

    assert count == 2
    assert 'id="odds-1_01">3.2<' in enriched
    assert 'id="ninki-1_01">1<' in enriched
    assert 'id="odds-1_02">8.6<' in enriched
    assert 'id="ninki-1_02">2<' in enriched
    assert NetkeibaHtmlCollector._has_embedded_win_odds(enriched)


def test_pedigree_parser():
    parsed = NetkeibaDatabaseCreator._parse_pedigree_html(
        PEDIGREE_HTML
    )
    assert parsed == {
        "sire_id": "sire001",
        "sire_name": "父馬",
        "dam_id": "dam001",
        "dam_name": "母馬",
        "damsire_id": "damsire001",
        "damsire_name": "母父馬",
    }


def test_extract_current_group():
    html = """
    <ul>
      <li date="20260725" group="1020260725"></li>
      <li class="Active"
          date="20260726"
          group="1020260725"></li>
    </ul>
    """
    assert (
        NetkeibaHtmlCollector._extract_current_group(
            html,
            "20260726",
        )
        == "1020260725"
    )
