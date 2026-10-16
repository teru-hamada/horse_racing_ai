# 登録済みの複数生成器を、特徴量セットの定義に従って実行・結合する。
# 各生成器のキー・出力列・対象行の順序を検査し、同じ出走馬の特徴量として安全に横結合する。
#
from __future__ import annotations

import pandas as pd

from .base import FEATURE_ID_COLUMNS
from .context import FeatureContext
from .registry import FeatureRegistry, FeatureSetDefinition


# 複数の生成器を順番に実行し、対象馬の順序と出力列の契約を検証して結合する。
class FeaturePipeline:
    # 生成器を解決するレジストリを受け取り、後続のtransformで利用する。
    def __init__(self, registry: FeatureRegistry) -> None:
        self.registry = registry

    # 各生成器の出力を検査し、targetsの行順・キーを保ったまま特徴量列を結合する。
    def transform(
        self,
        targets: pd.DataFrame,
        history: pd.DataFrame,
        definition: FeatureSetDefinition,
    ) -> pd.DataFrame:
        # targets: 特徴量を付与する対象行。race_id＋horse_idと並び順を維持する。
        # history: 履歴の入力表。集計対象の日付制限はこの処理の境界条件に従う。
        missing = set(FEATURE_ID_COLUMNS).difference(targets.columns)
        # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
        if missing:
            raise ValueError(f"Targets are missing feature IDs: {sorted(missing)}")

        context = FeatureContext(
            feature_set_name=definition.name,
            feature_set_version=definition.version,
            parameters=definition.parameters,
        )
        result = targets.loc[:, FEATURE_ID_COLUMNS].reset_index(drop=True).copy()
        for generator in self.registry.resolve(definition):
            generated = generator.transform(targets, history, context).reset_index(drop=True)
            generator.validate_output(targets.reset_index(drop=True), generated)
            result = pd.concat(
                [result, generated.loc[:, generator.output_columns]],
                axis=1,
            )
        return result
