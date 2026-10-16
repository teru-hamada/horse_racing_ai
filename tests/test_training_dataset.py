# 検証対象：任意の特徴量セットの結合、欠損フラグ、目的変数の入力除外、列名衝突の拒否を確認する。
# 明示した日付境界で学習・検証・テストへ分割されることを検証する。
#
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from src.public_api import (
    TrainingDatasetBuilder,
    TrainingDatasetConfig,
    save_features,
)


# テストで結合対象に使う、小さなID・日付・正解ラベルの表を用意する。
def _base() -> pd.DataFrame:
    return pd.DataFrame([
        {"race_id": "r1", "horse_id": "h1", "race_date": "2026-01-01", "target_top3": 1},
        {"race_id": "r2", "horse_id": "h1", "race_date": "2026-02-01", "target_top3": 0},
        {"race_id": "r3", "horse_id": "h2", "race_date": "2026-03-01", "target_top3": 1},
    ])


# セット名・版・生成runなどの管理列を持つ特徴量表を作り、保存・結合の検証へ渡す。
def _features(name: str, column: str, values: list[float]) -> pd.DataFrame:
    base = _base().iloc[:len(values)]
    return pd.DataFrame({
        "race_id": base["race_id"],
        "horse_id": base["horse_id"],
        "race_date": base["race_date"],
        "feature_set_name": name,
        "feature_set_version": "1.0.0",
        "history_cutoff": pd.to_datetime(base["race_date"]),
        "feature_run_id": f"{name}-run",
        "generated_at": datetime(2026, 4, 1),
        column: values,
    })


# 鮮度を固定でfreshにする代替チェッカー。対象のテストでは結合ロジックだけを検証する。
def _fresh(**kwargs):
    return {"status": "fresh"}


# 異なるセットを左結合し、欠損フラグを付けながら正解列を入力Xから除外する。
def test_builder_joins_arbitrary_feature_sets_without_target_leakage(tmp_path):
    # このテスト専用の特徴量DBパス。本番のfeatures.duckdbは使わない。
    feature_db = tmp_path / "features.duckdb"
    save_features(_features("baseline", "prior_starts", [0.0, 1.0, 0.0]), feature_db)
    save_features(_features("jockey_index", "jockey_score", [70.0, 80.0]), feature_db)
    builder = TrainingDatasetBuilder(
        feature_database=feature_db,
        freshness_checkers={
            "baseline:1.0.0": _fresh,
            "jockey_index:1.0.0": _fresh,
        },
    )

    result = builder.build(_base(), TrainingDatasetConfig(
        feature_sets=("baseline:1.0.0", "jockey_index:1.0.0"),
    ))

    assert result.feature_columns == (
        "prior_starts", "jockey_score__missing", "jockey_score"
    )
    assert "target_top3" not in result.X.columns
    assert result.reports[1].missing_rows == 1
    assert result.frame.loc[2, "jockey_score__missing"] == 1
    assert result.manifest["feature_sets"][1]["identifier"] == "jockey_index:1.0.0"


# 別セットの同名特徴量を、曖昧な列名のまま結合せず例外にする。
def test_builder_rejects_colliding_feature_column_names(tmp_path):
    # このテスト専用の特徴量DBパス。本番のfeatures.duckdbは使わない。
    feature_db = tmp_path / "features.duckdb"
    save_features(_features("one", "shared_score", [1.0, 2.0, 3.0]), feature_db)
    save_features(_features("two", "shared_score", [4.0, 5.0, 6.0]), feature_db)
    builder = TrainingDatasetBuilder(
        feature_database=feature_db,
        freshness_checkers={"one:1.0.0": _fresh, "two:1.0.0": _fresh},
    )

    with pytest.raises(ValueError, match="collide"):
        builder.build(_base(), TrainingDatasetConfig(
            feature_sets=("one:1.0.0", "two:1.0.0"),
        ))


# 指定した検証開始日・テスト開始日に従い、各日が一つの区分へ入る。
def test_builder_splits_on_explicit_date_boundaries(tmp_path):
    # このテスト専用の特徴量DBパス。本番のfeatures.duckdbは使わない。
    feature_db = tmp_path / "features.duckdb"
    save_features(_features("baseline", "prior_starts", [0.0, 1.0, 0.0]), feature_db)
    builder = TrainingDatasetBuilder(
        feature_database=feature_db,
        freshness_checkers={"baseline:1.0.0": _fresh},
    )
    config = TrainingDatasetConfig(
        feature_sets=("baseline:1.0.0",),
        validation_start_date=date(2026, 2, 1),
        test_start_date=date(2026, 3, 1),
    )
    # 学習表と選択特徴量・結合レポート・構成記録をひとまとめにした結果。
    dataset = builder.build(_base(), config)

    splits = builder.split_by_date(dataset, config)

    assert {name: len(part.frame) for name, part in splits.items()} == {
        "train": 1, "validation": 1, "test": 1,
    }
