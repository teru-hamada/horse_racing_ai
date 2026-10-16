# netkeiba共通実装をHTML収集専用の入口として公開する。
# ここで行うのは取得とキャッシュ保存であり、保存済みHTMLのDB用解析とは切り離す。
#
from __future__ import annotations

from datetime import date
from importlib import import_module


_common = import_module("src.00_common.netkeiba_common")
# netkeibaのページ要求間隔（秒）。連続アクセスを抑える共通値。
PAGE_REQUEST_INTERVAL_SECONDS = _common.PAGE_REQUEST_INTERVAL_SECONDS
NetkeibaCommon = _common.NetkeibaCommon


# 共通スクレイパーのうちHTML収集の操作を公開する用途別クラス。
class NetkeibaHtmlCollector(NetkeibaCommon):
    """netkeibaからHTMLを取得し、ファイルへ保存する。"""

    # 指定期間のHTML収集を共通実装へ委ね、日付・レース・馬・オッズの取得件数を返す。
    def collect_html_date_range(
        self,
        start_date: date,
        end_date: date,
        dataset_type: str,
        force: bool = False,
        progress_callback=None,
    ) -> dict[str, int]:
        # force: キャッシュを使わず新規取得するかを指定する。
        return self._collect_html_date_range(
            start_date=start_date,
            end_date=end_date,
            dataset_type=dataset_type,
            force=force,
            progress_callback=progress_callback,
        )
