# src/40_ai_modeling/feature_engineering/commonをPythonパッケージとして認識させる初期化ファイル。
# 利用側が必要な生成器・設定・集計関数をまとめてimportできるよう、公開名をここで再公開する。
#
"""Reusable calculations shared by feature generators."""

from .time_decay import (
    calculate_elapsed_days,
    decay_weight,
    decay_weights,
    effective_count,
    weighted_mean,
    weighted_sum,
)

# このモジュールから明示的に公開する名前。実装内部の補助名と区別する。
__all__ = [
    "calculate_elapsed_days",
    "decay_weight",
    "decay_weights",
    "effective_count",
    "weighted_mean",
    "weighted_sum",
]
