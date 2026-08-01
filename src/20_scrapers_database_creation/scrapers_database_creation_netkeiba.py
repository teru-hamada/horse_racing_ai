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
