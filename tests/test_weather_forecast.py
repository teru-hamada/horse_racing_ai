# 検証対象：天気と馬場の対応、対象日の正午に近い時刻の選択、日別予報への代替を確認する。
# 偽のHTTPセッションで競馬場別URLと応答解析を検証し、実際の予報APIには接続しない。
#
from __future__ import annotations

from datetime import date

import pytest

from src.public_api import (
    fetch_jravan_weather,
    parse_jravan_forecast,
    weather_to_track_condition,
)


# 天気カテゴリごとの既定馬場状態が、対応表どおりになることを確認する。
@pytest.mark.parametrize(
    ("weather", "track_condition"),
    [
        ("晴", "良"),
        ("曇", "良"),
        ("小雨", "稍重"),
        ("雨", "重"),
        ("小雪", "稍重"),
        ("雪", "不良"),
    ],
)
def test_weather_to_track_condition_mapping(weather, track_condition):
    assert weather_to_track_condition(weather) == track_condition


# 別日の予報を除き、対象日の時刻別予報で正午に最も近い情報を選ぶ。
def test_forecast_uses_target_date_hour_nearest_noon():
    payload = {"wxdata": [{
        "srf": [
            {"date": "2026-09-06T09:00:00+09:00", "wx": 100},
            {"date": "2026-09-06T12:00:00+09:00", "wx": 300},
            {"date": "2026-09-07T12:00:00+09:00", "wx": 200},
        ],
        "mrf": [{"date": "2026-09-06T00:00:00+09:00", "wx": 200}],
    }]}

    result = parse_jravan_forecast(payload, date(2026, 9, 6))

    assert result == {
        "weather": "雨",
        "track_condition": "重",
        "forecast_at": "2026-09-06T12:00:00+09:00",
        "forecast_type": "hourly",
    }


# 対象日の時刻別予報が使えない場合、同日の日別予報へ切り替える。
def test_forecast_falls_back_to_target_date_daily_data():
    payload = {"wxdata": [{
        "srf": [],
        "mrf": [{"date": "2026-09-13T00:00:00+09:00", "wx": 200}],
    }]}

    result = parse_jravan_forecast(payload, date(2026, 9, 13))

    assert result["weather"] == "曇"
    assert result["track_condition"] == "良"
    assert result["forecast_type"] == "daily"


# 競馬場に対応する公開予報URLを利用し、取得した天気と馬場推定を返す。
def test_fetch_uses_jravan_venue_forecast_url():
    # HTTP応答の代替。本文やJSONなど、テスト対象が参照する属性・メソッドを提供する。
    class Response:
        # 偽HTTP応答の状態検査。実装したエラー条件だけをテスト対象へ通知する。
        def raise_for_status(self):
            return None

        # 偽HTTP応答のJSON本文を返し、API解析へ決められた入力を与える。
        def json(self):
            return {"wxdata": [{
                "srf": [{"date": "2026-09-06T12:00:00+09:00", "wx": 400}],
                "mrf": [],
            }]}

    # HTTPセッションの代替実装。実通信を行わず、用意した応答を返す。
    class Session:
        # テスト用の入力・応答・呼出履歴を初期化し、ケースごとに独立した状態を用意する。
        def __init__(self):
            self.url = None

        # GET要求を実通信の代わりに受け、用意した応答を返す。
        def get(self, url, **kwargs):
            self.url = url
            return Response()

    session = Session()
    result = fetch_jravan_weather(
        "中京", date(2026, 9, 6), session=session
    )

    assert session.url.endswith("/weather/data/chukyo/forecast/latest.json")
    assert result["weather"] == "雪"
    assert result["track_condition"] == "不良"
