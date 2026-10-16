# 登録済みタスクに学習・予想・過去レース予想・日別予想を振り分ける。
# 既存のTop3向けAPIも残しながら、タスク名を指定する共通APIへ接続する。
#
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
from .tasks.top3.model import predict_race_date as _predict_top3_date
from .tasks.top3.train import train_model as _train_top3


# 登録を確認したタスクの学習実装へ渡し、学習指標を返す。
def train_task(
    task_name: str,
    historical_records: pd.DataFrame,
    config: TrainConfig,
    log: Callable[[str], None],
    progress: Callable[[float], None] | None = None,
) -> dict[str, object]:
    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # progress: 任意の進捗コールバック。値は通常0～1の完了率。
    get_task(task_name)
    if task_name == "top3":
        return _train_top3(
            historical_records,
            config,
            log,
            progress,
        )
    raise AssertionError(f"タスク実装がありません: {task_name}")


# タスク名を検証してから、指定レースの予想実装へ振り分ける。
def predict_task(
    task_name: str,
    historical_records: pd.DataFrame,
    upcoming_records: pd.DataFrame,
    race_id: str,
    model_dir: Path,
):
    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # upcoming_records: 予想対象の出馬表。結果がまだ分からない行として扱う。
    # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
    # model_dir: 重み・前処理器・指標が同じ学習runとして保存されたディレクトリ。
    get_task(task_name)
    if task_name == "top3":
        return _predict_top3(
            historical_records,
            upcoming_records,
            race_id,
            model_dir,
        )
    raise AssertionError(f"タスク実装がありません: {task_name}")


# 指定タスクの過去レース予想を呼び出し、確定結果との比較に使う。
def predict_historical_task(
    task_name: str,
    historical_records: pd.DataFrame,
    race_id: str,
    model_dir: Path,
):
    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
    # model_dir: 重み・前処理器・指標が同じ学習runとして保存されたディレクトリ。
    get_task(task_name)
    if task_name == "top3":
        return _predict_top3_historical(
            historical_records,
            race_id,
            model_dir,
        )
    raise AssertionError(f"タスク実装がありません: {task_name}")


# 指定タスクの開催日一括予想へ、履歴・出馬表・モデル保存先を渡す。
def predict_date_task(
    task_name: str,
    historical_records: pd.DataFrame,
    upcoming_records: pd.DataFrame,
    race_date: object,
    model_dir: Path,
):
    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # upcoming_records: 予想対象の出馬表。結果がまだ分からない行として扱う。
    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    # model_dir: 重み・前処理器・指標が同じ学習runとして保存されたディレクトリ。
    get_task(task_name)
    if task_name == "top3":
        return _predict_top3_date(
            historical_records, upcoming_records, race_date, model_dir
        )
    raise AssertionError(f"タスク実装がありません: {task_name}")


# 既存画面の学習呼出しをTop3タスクへ接続する互換窓口。タスク選択は呼出側へ要求しない。
def train_model(
    historical_records: pd.DataFrame,
    config: TrainConfig,
    log: Callable[[str], None],
    progress: Callable[[float], None] | None = None,
) -> dict[str, object]:
    """現行画面向けのTop3学習互換窓口。"""
    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # progress: 任意の進捗コールバック。値は通常0～1の完了率。
    return train_task("top3", historical_records, config, log, progress)


# 既存の1レース予想呼出しをTop3の共通タスクAPIへ渡す。
def predict_race(*args, **kwargs):
    """現行画面向けのTop3予想互換窓口。"""
    return predict_task("top3", *args, **kwargs)


# 既存の過去レース予想呼出しをTop3タスクへ接続する。
def predict_historical_race(*args, **kwargs):
    """現行画面向けのTop3過去予想互換窓口。"""
    return predict_historical_task("top3", *args, **kwargs)


# 画面からの日付一括予想をTop3実装へ委ね、引数と戻り値の互換性を保つ。
def predict_race_date(*args, **kwargs):
    """現行画面向けのTop3開催日一括予想窓口。"""
    return predict_date_task("top3", *args, **kwargs)


# このモジュールから明示的に公開する名前。実装内部の補助名と区別する。
__all__ = [
    "MODEL_TASKS",
    "TrainConfig",
    "train_task",
    "predict_task",
    "predict_historical_task",
    "predict_date_task",
    "train_model",
    "predict_race",
    "predict_historical_race",
    "predict_race_date",
]
