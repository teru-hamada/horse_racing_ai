# 過去レースの学習用表に、3着以内なら1となる目的変数を付ける。
# 履歴特徴量の生成と着順の有効性確認を経た表を利用し、特徴量と正解ラベルを区別する。
#
from collections.abc import Callable

import pandas as pd

from ...common.features import build_historical_training_base


# 履歴特徴量と有効な着順を持つ表へ、3着以内を1とする正解列を追加する。
def build_training_frame(
    records: pd.DataFrame,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """3着以内を1とする学習データを作る。"""
    featured = build_historical_training_base(records, log=log)
    featured["target_top3"] = (
        featured["finish_position"] <= 3
    ).astype(int)
    return featured
