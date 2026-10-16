# 1回の特徴量生成で共有するセット名・版・パラメーターを保持する。
# 生成器ごとの設定を同じコンテキストから参照し、どのレシピで計算したかを識別する。
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


# セット名・版・計算パラメーターを生成器へ渡す共通のコンテキスト。
@dataclass(frozen=True)
class FeatureContext:
    """Immutable settings shared by all generators in one feature run."""

    # 予想入力用の特徴量セット名。版と組み合わせて保存データを識別する。
    feature_set_name: str
    # 特徴量セットの版。計算定義が異なるデータを区別して保存する。
    feature_set_version: str
    # 生成器名ごとの計算パラメーター。生成器は自身に対応する設定を参照する。
    parameters: Mapping[str, Any] = field(default_factory=dict)

    # 名前と版をコロンで連結した識別子。保存・設定・照合で同じ形式を使う。
    @property
    def identifier(self) -> str:
        return f"{self.feature_set_name}:{self.feature_set_version}"
