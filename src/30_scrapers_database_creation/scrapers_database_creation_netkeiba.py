# 保存済みnetkeiba HTMLの解析専用の入口。
# 共通パーサーを呼び出してDB登録用の表を返し、HTML収集処理との責務を分ける。
#
from __future__ import annotations

from datetime import date
from importlib import import_module

import pandas as pd


NetkeibaCommon = import_module(
    "src.00_common.netkeiba_common"
).NetkeibaCommon


# 保存済みHTMLの解析を公開する用途別クラス。新しいHTMLの取得とは分離する。
class NetkeibaDatabaseCreator(NetkeibaCommon):
    """保存済みnetkeiba HTMLを解析し、DB保存用データを作る。"""

    # 保存済みHTMLを解析する共通処理へ条件を渡し、DB保存用の表を返す。
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
