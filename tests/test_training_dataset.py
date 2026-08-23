from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from src.public_api import (
    TrainingDatasetBuilder,
    TrainingDatasetConfig,
    save_features,
)


def _base() -> pd.DataFrame:
    return pd.DataFrame([
        {"race_id": "r1", "horse_id": "h1", "race_date": "2026-01-01", "target_top3": 1},
        {"race_id": "r2", "horse_id": "h1", "race_date": "2026-02-01", "target_top3": 0},
        {"race_id": "r3", "horse_id": "h2", "race_date": "2026-03-01", "target_top3": 1},
    ])


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


def _fresh(**kwargs):
    return {"status": "fresh"}


def test_builder_joins_arbitrary_feature_sets_without_target_leakage(tmp_path):
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


def test_builder_rejects_colliding_feature_column_names(tmp_path):
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


def test_builder_splits_on_explicit_date_boundaries(tmp_path):
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
    dataset = builder.build(_base(), config)

    splits = builder.split_by_date(dataset, config)

    assert {name: len(part.frame) for name, part in splits.items()} == {
        "train": 1, "validation": 1, "test": 1,
    }
