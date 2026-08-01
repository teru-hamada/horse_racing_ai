from __future__ import annotations

from datetime import date
from importlib import import_module


_common = import_module("src.00_common.netkeiba_common")
PAGE_REQUEST_INTERVAL_SECONDS = _common.PAGE_REQUEST_INTERVAL_SECONDS
NetkeibaCommon = _common.NetkeibaCommon


class NetkeibaHtmlCollector(NetkeibaCommon):
    """netkeibaからHTMLを取得し、ファイルへ保存する。"""

    def collect_html_date_range(
        self,
        start_date: date,
        end_date: date,
        dataset_type: str,
        force: bool = False,
        progress_callback=None,
    ) -> dict[str, int]:
        return self._collect_html_date_range(
            start_date=start_date,
            end_date=end_date,
            dataset_type=dataset_type,
            force=force,
            progress_callback=progress_callback,
        )
