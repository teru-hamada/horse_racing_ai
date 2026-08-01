from collections.abc import Callable

import pandas as pd

from ...common.features import build_historical_training_base


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
