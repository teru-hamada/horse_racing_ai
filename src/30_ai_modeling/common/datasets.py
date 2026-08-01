from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """開催日順に学習70%、検証15%、テスト15%へ分割する。"""
    dates = np.array(
        sorted(pd.to_datetime(frame["race_date"]).dropna().unique())
    )
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
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "学習・検証・テストのいずれかが0件です。"
            "データ期間を増やしてください。"
        )
    return train, validation, test
