from __future__ import annotations

from datetime import date

import pandas as pd

from src.public_api import (
    RecentFormRunConfig,
    feature_runs,
    generate_recent_form_features,
    load_features,
)


def _history() -> pd.DataFrame:
    rows = []
    for race_number, race_date in enumerate(
        ["2026-01-01", "2026-01-02", "2026-01-03"], start=1
    ):
        for horse_number in range(1, 4):
            rows.append({
                "race_id": f"r{race_number}",
                "horse_id": f"h{horse_number}",
                "race_date": race_date,
                "finish_position": horse_number,
                "last_3f_time": 35.0 + horse_number,
                "collected_at": "2026-01-04",
            })
    return pd.DataFrame(rows)


def test_service_generates_requested_period_and_records_run(tmp_path, monkeypatch):
    from importlib import import_module

    service = import_module("src.30_ai_modeling.feature_engineering.service")
    monkeypatch.setattr(service._source_storage, "load_records", lambda _: _history())
    feature_db = tmp_path / "features.duckdb"
    config = RecentFormRunConfig(
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 3),
    )

    result = generate_recent_form_features(config, feature_database=feature_db)
    features = load_features("baseline", "1.1.0", feature_db)
    runs = feature_runs(feature_db)

    assert result["status"] == "completed"
    assert result["row_count"] == 6
    assert len(features) == 6
    assert features["race_date"].min() == pd.Timestamp("2026-01-02")
    assert runs["status"].iloc[0] == "completed"
    assert len(runs["source_data_fingerprint"].iloc[0]) == 64
    first_day = features[features["race_date"].eq(pd.Timestamp("2026-01-02"))]
    assert (first_day["recent_starts_total"] == 1).all()
