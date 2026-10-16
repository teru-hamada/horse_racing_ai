# 開催日の時系列順に学習・検証・テストへ分割する。
# 同じ日のレースを区分間で分けず、過去で学習して将来で評価する順序を維持する。
#
from __future__ import annotations

import numpy as np
import pandas as pd


# 異なる開催日を基準に70%・15%・15%へ分割する。同じ日の行を区分間で分けない。
def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """開催日順に学習70%、検証15%、テスト15%へ分割する。"""
    # 70%・15%・15%は行数ではなく異なる開催日数を基準とするため、各区分の行数比は変わり得る。
    dates = np.array(
        sorted(pd.to_datetime(frame["race_date"]).dropna().unique())
    )
    # 三分割を安定して作れるだけの異なる開催日数がなければ、データ追加を要求する。
    if len(dates) < 8:
        raise ValueError(
            "時系列分割には少なくとも8開催日分の"
            "履歴データが必要です。"
        )
    train_end = max(1, int(len(dates) * 0.70))
    validation_end = max(train_end + 1, int(len(dates) * 0.85))
    date_series = pd.to_datetime(frame["race_date"])
    train = frame[date_series.isin(set(dates[:train_end]))].copy()
    validation = frame[
        date_series.isin(set(dates[train_end:validation_end]))
    ].copy()
    test = frame[date_series.isin(set(dates[validation_end:]))].copy()
    # 学習・検証・テストが一つでも空なら、学習や指標算出へ進まない。
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "学習・検証・テストのいずれかが0件です。"
            "データ期間を増やしてください。"
        )
    return train, validation, test
