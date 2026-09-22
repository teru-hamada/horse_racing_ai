from __future__ import annotations
from importlib import import_module as _import_module

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

JraResultFetcher = _import_module('src.20_scrapers_html_collection.jra_results').JraResultFetcher
parse_jra_result_html = _import_module('src.20_scrapers_html_collection.jra_results').parse_jra_result_html


RESULT_HTML = """<!doctype html><html><head><meta charset="Shift_JIS"></head><body>
<table class="basic striped"><thead><tr><th class="place">着順</th></tr></thead>
<tbody>
<tr><td class="place">1</td><td class="waku"><img alt="枠2黒"></td>
<td class="num">3</td><td class="horse">勝ち馬</td></tr>
<tr><td class="place">2</td><td class="waku"><img alt="枠4青"></td>
<td class="num">8</td><td class="horse">二着馬</td></tr>
</tbody></table>
<div class="refund_area">
<li class="win"><div class="line"><div class="num">3</div><div class="yen">360円</div></div></li>
<li class="place"><div class="line"><div class="num">3</div><div class="yen">150円</div></div></li>
<li class="umaren"><div class="line"><div class="num">3-8</div><div class="yen">930円</div></div></li>
<li class="umatan"><div class="line"><div class="num">3-8</div><div class="yen">1,690円</div></div></li>
</div></body></html>"""


@pytest.mark.parametrize("race_id,horse_number,status", [
    ("202609040704", 7, "excluded"),
    ("202609040706", 15, "did_not_finish"),
])
def test_actual_september_results_preserve_non_finishers(race_id, horse_number, status):
    frame = parse_jra_result_html(Path(__file__).parent / "fixtures/jra" / f"{race_id}.html", race_id)
    horse = frame.set_index("horse_number").loc[horse_number]
    assert horse.result_status == status
    assert pd.isna(horse.finish_position)
    assert frame.finish_position.eq(1).sum() == 1
    assert frame.result_status.eq("finished").sum() == len(frame) - 1
    assert frame.attrs["official_refunds"] == {
        "horse_numbers": [7] if status == "excluded" else [],
        "frame_numbers": [],
        "same_frame_numbers": [4] if status == "excluded" else [],
    }


@pytest.mark.parametrize("label,status", [
    ("取消", "scratched"), ("失格", "disqualified"),
    ("未確定", "unknown"), ("", "unknown"), ("不明1", "unknown"),
])
def test_non_numeric_finish_is_never_guessed(tmp_path, label, status):
    path = tmp_path / "result.html"
    path.write_text(RESULT_HTML.replace('<td class="place">2</td>', f'<td class="place">{label}</td>'), encoding="utf-8")
    frame = parse_jra_result_html(path, "r1")
    assert frame.iloc[1].result_status == status
    assert pd.isna(frame.iloc[1].finish_position)


@pytest.mark.parametrize("refund_text,expected", [
    ("返還馬番　３番、８番　返還枠番　２枠　返還同枠　４枠", {
        "horse_numbers": [3, 8], "frame_numbers": [2], "same_frame_numbers": [4],
    }),
    ("返還馬番 3番 返還対象不明", None),
    ("返還馬番 3番 返還同枠", None),
    ("返還馬番 0番", None),
])
def test_official_refund_sections_are_parsed_or_rejected(tmp_path, refund_text, expected):
    path = tmp_path / "result.html"
    html = RESULT_HTML.replace('<div class="refund_area">',
        '<div class="refund_area"><div class="restoration"><dl><dt>返還</dt>'
        f'<dd>{refund_text}</dd></dl></div>')
    path.write_text(html, encoding="utf-8")
    if expected is None:
        with pytest.raises(ValueError, match="公式返還"):
            parse_jra_result_html(path, "r1")
    else:
        assert parse_jra_result_html(path, "r1").attrs["official_refunds"] == expected


class _Logger:
    def info(self, message: str) -> None:
        pass


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.encoding = None

    def raise_for_status(self) -> None:
        pass


class _Session:
    def __init__(self) -> None:
        self.headers = {}
        self.calls: list[str] = []

    def post(self, url, data, headers, timeout):
        cname = data["cname"]
        self.calls.append(cname)
        pages = {
            JraResultFetcher.ENTRY_CNAME: (
                "<a onclick=\"doAction('/JRADB/accessS.html', "
                "'pw01srl10052026030320260613/DC')\">東京</a>"
            ),
            "pw01srl10052026030320260613/DC": (
                '<a href="/JRADB/accessS.html?CNAME='
                'pw01sde1005202603031220260613/BE">12R</a>'
            ),
            "pw01sde1005202603031220260613/BE": RESULT_HTML,
        }
        return _Response(pages[cname])


def test_parse_jra_result_html_reads_finish_and_official_payouts(tmp_path):
    path = tmp_path / "result.html"
    path.write_text(RESULT_HTML, encoding="utf-8")

    result = parse_jra_result_html(path, "202605030312")

    assert result[["horse_number", "frame_number", "finish_position"]].to_dict(
        "records"
    ) == [
        {"horse_number": 3, "frame_number": 2, "finish_position": 1},
        {"horse_number": 8, "frame_number": 4, "finish_position": 2},
    ]
    assert result.attrs["official_payouts"] == {
        "win:3": 360,
        "place:3": 150,
        "quinella:3-8": 930,
        "exacta:3-8": 1690,
    }


def test_jra_result_fetcher_follows_official_links_and_saves_html(tmp_path):
    session = _Session()
    fetcher = JraResultFetcher(
        _Logger(), interval_seconds=0, session=session, output_root=tmp_path
    )

    result = fetcher.fetch_result_for_comparison(
        "202605030312", date(2026, 6, 13), force=True
    )

    assert result["finish_position"].tolist() == [1, 2]
    assert session.calls == [
        JraResultFetcher.ENTRY_CNAME,
        "pw01srl10052026030320260613/DC",
        "pw01sde1005202603031220260613/BE",
    ]
    saved = tmp_path / "2026" / "result" / "202605030312.html"
    assert saved.is_file()
    assert 'charset="UTF-8"' in saved.read_text(encoding="utf-8")

    offline_session = _Session()
    offline_session.post = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("保存済みHTMLの再利用時に通信してはいけません")
    )
    cached = JraResultFetcher(
        _Logger(), interval_seconds=0, session=offline_session,
        output_root=tmp_path,
    ).fetch_result_for_comparison(
        "202605030312", date(2026, 6, 13), force=False
    )
    assert cached["finish_position"].tolist() == [1, 2]
