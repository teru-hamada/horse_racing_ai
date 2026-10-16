# 特徴量生成器を名前で登録し、セットの版・パラメーター・出力列を解決する。
# 生成器間の列名衝突を検出し、同名列の上書きや意図しない結合を防ぐ。
#
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .base import FeatureGenerator


# セット名・版として利用可能な文字列の形式。空値や危険な区切り文字を拒否する。
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# どの生成器とパラメーターで一つの特徴量セットを作るかを表す版付きレシピ。
@dataclass(frozen=True)
class FeatureSetDefinition:
    """Versioned recipe describing exactly which generators produced a set."""

    # 登録・保存に使う内部名。画面ラベルとは別に扱う。
    name: str
    # この生成器または定義の版。計算内容を変更するときの識別に使う。
    version: str
    # このセットを構成する生成器名の順序付き一覧。
    generators: tuple[str, ...]
    # 生成器名ごとの計算パラメーター。生成器は自身に対応する設定を参照する。
    parameters: Mapping[str, Any] = field(default_factory=dict)

    # 初期化直後に名前・版・構成の整合性を検査し、不正な設定を計算開始前に拒否する。
    def __post_init__(self) -> None:
        if not _VERSION_PATTERN.fullmatch(self.name):
            raise ValueError(f"Invalid feature set name: {self.name!r}")
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ValueError(f"Invalid feature set version: {self.version!r}")
        if len(set(self.generators)) != len(self.generators):
            raise ValueError("Feature generator names must be unique")

    # 名前と版をコロンで連結した識別子。保存・設定・照合で同じ形式を使う。
    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"


# 生成器名から実装を引く登録簿。同名登録や特徴量列の衝突を登録段階で拒否する。
class FeatureRegistry:
    # 空の登録簿を作り、初期生成器を通常のregisterと同じ検査を通して追加する。
    def __init__(self, generators: Iterable[FeatureGenerator] = ()) -> None:
        self._generators: dict[str, FeatureGenerator] = {}
        for generator in generators:
            self.register(generator)

    # 登録済み生成器の出力列との衝突を検査してから追加する。
    def register(self, generator: FeatureGenerator) -> None:
        if not generator.name or not _VERSION_PATTERN.fullmatch(generator.version):
            raise ValueError("Generator name and version must be non-empty and version-safe")
        if generator.name in self._generators:
            raise ValueError(f"Feature generator already registered: {generator.name}")
        duplicated = set(generator.output_columns).intersection(self.output_columns)
        if duplicated:
            raise ValueError(f"Feature columns already registered: {sorted(duplicated)}")
        self._generators[generator.name] = generator

    # 登録済み生成器が宣言した出力列の集合。新しい生成器の列名衝突を調べる。
    @property
    def output_columns(self) -> set[str]:
        return {
            column
            for generator in self._generators.values()
            for column in generator.output_columns
        }

    # セット定義で指定された生成器を、定義順に解決する。
    def resolve(self, definition: FeatureSetDefinition) -> tuple[FeatureGenerator, ...]:
        missing = set(definition.generators).difference(self._generators)
        # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
        if missing:
            raise ValueError(f"Unknown feature generators: {sorted(missing)}")
        return tuple(self._generators[name] for name in definition.generators)
