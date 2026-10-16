# 検証対象：生成器の版管理、出力列の衝突検出、対象行の順序維持という共通契約を確認する。
# 小さな生成器を使い、競馬固有の計算から独立してパイプラインの接続を検証する。
#
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.public_api import (
    FeatureGenerator,
    FeaturePipeline,
    FeatureRegistry,
    FeatureSetDefinition,
)


# パイプライン接続を検証する小さな生成器。競馬の集計ロジックに依存しない固定値を返す。
class SampleGenerator(FeatureGenerator):
    # 登録・保存に使う内部名。画面ラベルとは別に扱う。
    name = "sample"
    # この生成器または定義の版。計算内容を変更するときの識別に使う。
    version = "1"
    # transformが返す特徴量列の宣言。識別キーを除いた列集合を出力検証で確認する。
    output_columns = ("sample_score",)

    # パイプラインの契約検査用の出力を作る。正常生成器と順序を変える生成器で違反検出を確認する。
    def transform(self, targets, history, context):
        output = targets[["race_id", "horse_id"]].copy()
        output["sample_score"] = float(context.parameters["score"])
        return output


# 登録した生成器と版が解決され、宣言された列と対象キーだけを持つ結果になることを確認する。
def test_pipeline_uses_registered_versioned_generator():
    targets = pd.DataFrame({"race_id": ["r1"], "horse_id": ["h1"]})
    definition = FeatureSetDefinition(
        name="baseline",
        version="1.0.0",
        generators=("sample",),
        parameters={"score": 3.5},
    )
    pipeline = FeaturePipeline(FeatureRegistry([SampleGenerator()]))

    result = pipeline.transform(targets, pd.DataFrame(), definition)

    assert result.to_dict("records") == [
        {"race_id": "r1", "horse_id": "h1", "sample_score": 3.5}
    ]


# 異なる生成器で同じ出力列を宣言した場合、曖昧な上書きを登録時に拒否する。
def test_registry_rejects_duplicate_feature_columns():
    registry = FeatureRegistry([SampleGenerator()])
    try:
        registry.register(SampleGenerator())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Duplicate generator was accepted")


# 生成器が対象行を並べ替えた場合、別の馬へ値を結合する前にエラーで止める。
def test_pipeline_rejects_changed_target_order():
    # 対象行を意図的に並べ替え、出力順序の違反をパイプラインが検出できるかを調べる。
    class ReorderedGenerator(SampleGenerator):
        # 登録・保存に使う内部名。画面ラベルとは別に扱う。
        name = "reordered"

        # パイプラインの契約検査用の出力を作る。正常生成器と順序を変える生成器で違反検出を確認する。
        def transform(self, targets, history, context):
            output = super().transform(targets, history, context)
            return output.iloc[::-1].reset_index(drop=True)

    targets = pd.DataFrame(
        {"race_id": ["r1", "r2"], "horse_id": ["h1", "h2"]}
    )
    pipeline = FeaturePipeline(FeatureRegistry([ReorderedGenerator()]))
    definition = FeatureSetDefinition("baseline", "1", ("reordered",), {"score": 1})

    try:
        pipeline.transform(targets, pd.DataFrame(), definition)
    except ValueError as exc:
        assert "preserve target row order" in str(exc)
    else:
        raise AssertionError("Reordered output was accepted")
