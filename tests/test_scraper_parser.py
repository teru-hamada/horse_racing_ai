from datetime import date

from src.logging_utils import AppLogger
from src.scrapers.netkeiba import NetkeibaScraper


RESULT_HTML = """
<html><head><title>2026年7月12日 福島11R</title></head><body>
<div class="RaceName">七夕賞</div>
<div class="RaceData01">芝2000m / 天候 : 晴 / 芝 : 良</div>
<div class="RaceData02">2回 福島 6日目</div>
<table class="RaceTable01">
<thead><tr><th>着順</th><th>枠</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th><th>騎手</th><th>タイム</th><th>単勝</th><th>人気</th><th>馬体重</th><th>厩舎</th></tr></thead>
<tbody>
<tr class="HorseList"><td>1</td><td>3</td><td>5</td><td><a href="/horse/2020100001">テストホース</a></td><td>牡4</td><td>57.0</td><td><a href="/jockey/01234">テスト騎手</a></td><td>1:59.8</td><td>3.2</td><td>1</td><td>480(+4)</td><td><a href="/trainer/01001">テスト厩舎</a></td></tr>
</tbody></table>
</body></html>
"""


def test_result_parser(tmp_path):
    logger = AppLogger(tmp_path)
    scraper = NetkeibaScraper(logger, interval_seconds=0.5)
    parsed = scraper._parse_page(RESULT_HTML, "202603020611", date(2026, 7, 12), "historical")
    assert len(parsed) == 1
    row = parsed.iloc[0]
    assert row["horse_id"] == "2020100001"
    assert row["horse_name"] == "テストホース"
    assert row["finish_position"] == 1
    assert row["surface"] == "芝"
    assert row["distance"] == 2000
    assert row["time_seconds"] == 119.8
