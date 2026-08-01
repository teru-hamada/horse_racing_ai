from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from .common.types import TrainConfig
from .registry import MODEL_TASKS, get_task
from .tasks.top3.predict import (
    predict_historical_race as _predict_top3_historical,
    predict_race as _predict_top3,
)
from .tasks.top3.train import train_model as _train_top3


def train_task(
    task_name: str,
    historical_records: pd.DataFrame,
    config: TrainConfig,
    log: Callable[[str], None],
    progress: Callable[[float], None] | None = None,
) -> dict[str, object]:
    get_task(task_name)
    if task_name == "top3":
        return _train_top3(
            historical_records,
            config,
            log,
            progress,
        )
    raise AssertionError(f"タスク実装がありません: {task_name}")


def predict_task(
    task_name: str,
    historical_records: pd.DataFrame,
    upcoming_records: pd.DataFrame,
    race_id: str,
    model_dir: Path,
):
    get_task(task_name)
    if task_name == "top3":
        return _predict_top3(
            historical_records,
            upcoming_records,
            race_id,
            model_dir,
        )
    raise AssertionError(f"タスク実装がありません: {task_name}")


def predict_historical_task(
    task_name: str,
    historical_records: pd.DataFrame,
    race_id: str,
    model_dir: Path,
):
    get_task(task_name)
    if task_name == "top3":
        return _predict_top3_historical(
            historical_records,
            race_id,
            model_dir,
        )
    raise AssertionError(f"タスク実装がありません: {task_name}")


def train_model(
    historical_records: pd.DataFrame,
    config: TrainConfig,
    log: Callable[[str], None],
    progress: Callable[[float], None] | None = None,
) -> dict[str, object]:
    """現行画面向けのTop3学習互換窓口。"""
    return train_task("top3", historical_records, config, log, progress)


def predict_race(*args, **kwargs):
    """現行画面向けのTop3予想互換窓口。"""
    return predict_task("top3", *args, **kwargs)


def predict_historical_race(*args, **kwargs):
    """現行画面向けのTop3過去予想互換窓口。"""
    return predict_historical_task("top3", *args, **kwargs)


__all__ = [
    "MODEL_TASKS",
    "TrainConfig",
    "train_task",
    "predict_task",
    "predict_historical_task",
    "train_model",
    "predict_race",
    "predict_historical_race",
]
