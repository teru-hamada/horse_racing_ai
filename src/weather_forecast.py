from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import requests


JRA_VAN_WEATHER_BASE_URL = "https://jra-van.jp/weather"
VENUE_IDS = {
    "札幌": "sapporo",
    "函館": "hakodate",
    "福島": "fukushima",
    "新潟": "niigata",
    "東京": "tokyo",
    "中山": "nakayama",
    "中京": "chukyo",
    "京都": "kyoto",
    "阪神": "hanshin",
    "小倉": "kokura",
}
WEATHER_TO_TRACK_CONDITION = {
    "晴": "良",
    "曇": "良",
    "小雨": "稍重",
    "雨": "重",
    "小雪": "稍重",
    "雪": "不良",
}


def weather_to_track_condition(weather: str) -> str:
    """Convert the displayed weather category to the requested default going."""

    try:
        return WEATHER_TO_TRACK_CONDITION[str(weather)]
    except KeyError as exc:
        raise ValueError(f"馬場状態へ変換できない天気です: {weather}") from exc


def weather_from_code(code: object) -> str | None:
    """Reduce the JRA-VAN forecast code to a model-compatible weather label."""

    try:
        family = int(code) // 100
    except (TypeError, ValueError):
        return None
    return {1: "晴", 2: "曇", 3: "雨", 4: "雪"}.get(family)


def _parse_datetime(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def parse_jravan_forecast(
    payload: dict[str, Any],
    target_date: date,
) -> dict[str, object] | None:
    """Use the target-date hourly forecast nearest noon, or the daily forecast."""

    wxdata = payload.get("wxdata")
    if not isinstance(wxdata, list) or not wxdata or not isinstance(wxdata[0], dict):
        return None
    forecast = wxdata[0]
    noon = datetime.combine(target_date, time(12))
    hourly_candidates: list[tuple[float, datetime, str]] = []
    for item in forecast.get("srf", []):
        if not isinstance(item, dict):
            continue
        forecast_time = _parse_datetime(item.get("date"))
        weather = weather_from_code(item.get("wx"))
        if forecast_time is None or forecast_time.date() != target_date or weather is None:
            continue
        distance = abs((forecast_time.replace(tzinfo=None) - noon).total_seconds())
        hourly_candidates.append((distance, forecast_time, weather))
    if hourly_candidates:
        _, forecast_time, weather = min(hourly_candidates, key=lambda item: item[0])
        return {
            "weather": weather,
            "track_condition": weather_to_track_condition(weather),
            "forecast_at": forecast_time.isoformat(),
            "forecast_type": "hourly",
        }

    for item in forecast.get("mrf", []):
        if not isinstance(item, dict):
            continue
        forecast_time = _parse_datetime(item.get("date"))
        weather = weather_from_code(item.get("wx"))
        if forecast_time is not None and forecast_time.date() == target_date and weather:
            return {
                "weather": weather,
                "track_condition": weather_to_track_condition(weather),
                "forecast_at": forecast_time.isoformat(),
                "forecast_type": "daily",
            }
    return None


def fetch_jravan_weather(
    course_name: str,
    target_date: date,
    *,
    timeout: int = 10,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Fetch the public forecast JSON used by the JRA-VAN weather page."""

    venue_id = VENUE_IDS.get(str(course_name))
    if venue_id is None:
        raise ValueError(f"JRA-VAN天気予報に対応していない競馬場です: {course_name}")
    source_url = (
        f"{JRA_VAN_WEATHER_BASE_URL}/data/{venue_id}/forecast/latest.json"
    )
    client = session or requests.Session()
    response = client.get(
        source_url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    parsed = parse_jravan_forecast(response.json(), target_date)
    if parsed is None:
        raise ValueError(
            f"{course_name}の{target_date.isoformat()}の天気予報がありません。"
        )
    return {**parsed, "course_name": course_name, "source_url": source_url}
