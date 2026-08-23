from __future__ import annotations

from datetime import date
from importlib import import_module

import pandas as pd


NetkeibaCommon = import_module(
    "src.00_common.netkeiba_common"
).NetkeibaCommon


class NetkeibaDatabaseCreator(NetkeibaCommon):
    """保存済みnetkeiba HTMLを解析し、DB保存用データを作る。"""

    def parse_cached_date_range(
        self,
        start_date: date,
        end_date: date,
        dataset_type: str,
        progress_callback=None,
    ) -> pd.DataFrame:
        return self._parse_cached_date_range(
            start_date=start_date,
            end_date=end_date,
            dataset_type=dataset_type,
            progress_callback=progress_callback,
        )

    def fetch_result_for_comparison(
        self,
        race_id: str,
        race_date: date,
        *,
        force: bool = True,
    ) -> pd.DataFrame:
        """Fetch and parse only the result HTML needed for prediction comparison."""

        url = self.RESULT_FALLBACK_URL.format(race_id=race_id)
        cache = self._cache_path(
            "historical", str(race_date.year), "result", race_id
        )
        self.logger.info(f"予想比較用の結果HTML取得対象URL: {url}")
        html = self._download(url, cache, force=force)
        frame = self._parse_page(
            html, race_id, race_date, dataset_type="historical"
        )
        if frame.empty or frame["finish_position"].notna().sum() == 0:
            raise ValueError(
                "確定着順を取得できませんでした。レース結果確定後に再実行してください。"
            )
        race_name = (
            frame["race_name"].dropna().iloc[0]
            if frame["race_name"].notna().any() else "レース名不明"
        )
        course_name = (
            frame["course_name"].dropna().iloc[0]
            if frame["course_name"].notna().any() else "開催地不明"
        )
        race_number = (
            frame["race_number"].dropna().iloc[0]
            if frame["race_number"].notna().any() else None
        )
        self._rename_race_html(
            cache, race_id, race_date, race_name, course_name, race_number
        )
        return frame
