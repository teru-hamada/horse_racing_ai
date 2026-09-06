from __future__ import annotations

from datetime import date

import pytest

from src.public_api import (
    fetch_jravan_weather,
    parse_jravan_forecast,
    weather_to_track_condition,
)


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


def test_forecast_falls_back_to_target_date_daily_data():
    payload = {"wxdata": [{
        "srf": [],
        "mrf": [{"date": "2026-09-13T00:00:00+09:00", "wx": 200}],
    }]}

    result = parse_jravan_forecast(payload, date(2026, 9, 13))

    assert result["weather"] == "曇"
    assert result["track_condition"] == "良"
    assert result["forecast_type"] == "daily"


def test_fetch_uses_jravan_venue_forecast_url():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"wxdata": [{
                "srf": [{"date": "2026-09-06T12:00:00+09:00", "wx": 400}],
                "mrf": [],
            }]}

    class Session:
        def __init__(self):
            self.url = None

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
