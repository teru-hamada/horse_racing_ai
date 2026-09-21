"""Fail-closed meeting-day detection from a complete monthly calendar."""
from calendar import monthrange
from datetime import date
import re

from bs4 import BeautifulSoup

from .date_collector import DateCollector


def parse_meeting_day(html: str, target: date) -> bool:
    soup = BeautifulSoup(html, "lxml")
    year = soup.select_one("#cal_select_year option[selected]")
    tables = soup.select("table.Calendar_Table")
    if year is None or year.get("value") != str(target.year) or len(tables) != 1:
        raise ValueError("開催カレンダーの年または表を確認できません。")
    days = {}
    month_links = []
    for cell in tables[0].select("td.RaceCellBox"):
        node = cell.select_one(".Day")
        if node is None:
            raise ValueError("開催カレンダーの日付構造が不正です。")
        day_text = node.get_text(strip=True)
        if not day_text:
            continue
        if not day_text.isascii() or not day_text.isdigit() or int(day_text) in days:
            raise ValueError("開催カレンダーの日付が不正または重複しています。")
        number = int(day_text)
        links = [match.group(1) for a in cell.select("a[href]")
                 if (match := re.search(r"kaisai_date=(\d{8})(?:&|$)", a["href"]))]
        if any(value != f"{target.year:04d}{target.month:02d}{number:02d}" for value in links):
            raise ValueError("開催カレンダーの対象月が一致しません。")
        month_links.extend(links)
        venues = cell.select(".JyoName")
        if any(not node.get_text(strip=True) for node in venues):
            raise ValueError("開催場の情報が欠落しています。")
        if (links or cell.select_one(".HaveData")) and not venues:
            raise ValueError("開催の表示と開催場が一致しません。")
        days[number] = bool(venues)
    if set(days) != set(range(1, monthrange(target.year, target.month)[1] + 1)) or not month_links:
        raise ValueError("対象月の完全な開催カレンダーを確認できません。")
    return days[target.day]


def check_meeting_day(target: date, output, logger) -> bool:
    collector = DateCollector(logger, output)
    try:
        path = output / f"calendar_{target:%Y%m}.html"
        html = collector._download(
            f"https://race.netkeiba.com/top/calendar.html?year={target.year}&month={target.month}",
            path, force=True,
        )
        return parse_meeting_day(html, target)
    finally:
        collector.session.close()
