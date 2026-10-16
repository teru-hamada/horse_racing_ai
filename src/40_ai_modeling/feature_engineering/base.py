# 特徴量生成器の共通契約と、出力する識別列を定義する。
# 生成器は名前・版・出力列を宣言し、入力の対象行と対応するrace_id＋horse_idを返す。
# 出力検証により、重複キーや宣言と異なる列構成をパイプラインへ流さない。
#
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .context import FeatureContext


# 生成器間で共通の結合キー。各出力で対象行と同じIDと順序を維持する。
FEATURE_ID_COLUMNS = ("race_id", "horse_id")


# 特徴量生成器の抽象基底。transformの実装と出力スキーマの宣言を各生成器へ要求する。
class FeatureGenerator(ABC):
    """Contract implemented by one independently versioned feature family."""

    # 登録・保存に使う内部名。画面ラベルとは別に扱う。
    name: str
    # この生成器または定義の版。計算内容を変更するときの識別に使う。
    version: str
    # transformが返す特徴量列の宣言。識別キーを除いた列集合を出力検証で確認する。
    output_columns: tuple[str, ...]

    # 生成器が実装する抽象メソッド。targetsと同じID・順序に、宣言した特徴量列を付けて返す。
    @abstractmethod
    def transform(
        self,
        targets: pd.DataFrame,
        history: pd.DataFrame,
        context: FeatureContext,
    ) -> pd.DataFrame:
        """Return ID columns plus this generator's declared output columns."""

    # 宣言済み出力列と識別キーを検査し、欠落や重複のある特徴量を後段へ渡さない。
    def validate_output(
        self,
        targets: pd.DataFrame,
        output: pd.DataFrame,
    ) -> None:
        # targets: 特徴量を付与する対象行。race_id＋horse_idと並び順を維持する。
        expected = set(FEATURE_ID_COLUMNS + self.output_columns)
        actual = set(output.columns)
        # 生成器が宣言した列と実際の列を完全一致で確認し、意図しない列の混入も拒否する。
        if actual != expected:
            raise ValueError(
                f"Feature generator {self.name!r} returned invalid columns: "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
        # 同一出走キーが複数あると横結合で行が増えるため、生成器の出力段階で拒否する。
        if output.duplicated(list(FEATURE_ID_COLUMNS)).any():
            raise ValueError(f"Feature generator {self.name!r} returned duplicate IDs")
        target_ids = targets.loc[:, FEATURE_ID_COLUMNS].reset_index(drop=True)
        output_ids = output.loc[:, FEATURE_ID_COLUMNS].reset_index(drop=True)
        # 件数だけでなくIDと順序を一致させ、別の馬に特徴量を結び付ける事故を防ぐ。
        if len(output_ids) != len(target_ids) or not output_ids.equals(target_ids):
            raise ValueError(
                f"Feature generator {self.name!r} must preserve target row order and IDs"
            )
