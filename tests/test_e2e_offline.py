from __future__ import annotations

import hashlib
from datetime import date, timedelta
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.public_api import (
    AppLogger,
    build_prediction_site,
    NetkeibaDatabaseCreator,
    RaceEntryRunConfig,
    RecentFormRunConfig,
    RecentSpeedRunConfig,
    SpeedIndexRunConfig,
    TrainConfig,
    generate_race_entry_features,
    generate_recent_form_features,
    generate_recent_speed_features,
    generate_speed_index_features,
    get_paths,
    load_records,
    predict_historical_race,
    predict_race_date,
    save_race_records,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _file_fingerprints(*roots: Path) -> dict[str, str]:
    """Capture protected artifacts so the test can prove they were untouched."""

    fingerprints: dict[str, str] = {}
    for root in roots:
        if root.is_file():
            files = [root]
        elif root.exists():
            files = sorted(path for path in root.rglob("*") if path.is_file())
        else:
            files = []
        for path in files:
            relative = str(path.relative_to(PROJECT_ROOT))
            fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def _offline_request(*args, **kwargs):
    raise AssertionError("E2Eテスト中の外部ネットワークアクセスは禁止されています")


def _parse_one_saved_result_html(log_dir: Path) -> pd.DataFrame:
    candidates = sorted(
        (PROJECT_ROOT / "data" / "raw_html" / "historical").glob(
            "**/result/*.html"
        )
    )
    assert candidates, "E2Eテストに使える保存済み結果HTMLがありません"
    html_path = candidates[0]
    race_id = html_path.name.split("_", 1)[0]
    creator = NetkeibaDatabaseCreator(AppLogger(log_dir), interval_seconds=0.5)
    parsed = creator._parse_page(
        html_path.read_text(encoding="utf-8", errors="replace"),
        race_id,
        date(2026, 1, 1),
        dataset_type="historical",
    )
    assert not parsed.empty
    assert parsed["finish_position"].notna().all()
    return parsed


def _minimum_training_history(parsed: pd.DataFrame, race_count: int = 48) -> pd.DataFrame:
    """Expand one real parsed field into the minimum chronological training set."""

    races = []
    first_date = date(2025, 1, 1)
    for index in range(race_count):
        race = parsed.copy()
        race["race_id"] = f"e2e_{index:04d}"
        race["race_date"] = first_date + timedelta(days=index)
        race["race_name"] = f"E2E検証レース{index + 1}"
        race["race_number"] = index % 12 + 1
        races.append(race)
    history = pd.concat(races, ignore_index=True)
    assert len(history) >= 500
    return history


@pytest.mark.e2e
def test_saved_html_to_prediction_comparison_is_offline_and_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    protected_before = _file_fingerprints(
        PROJECT_ROOT / "data" / "racing.duckdb",
        PROJECT_ROOT / "data" / "features.duckdb",
        PROJECT_ROOT / "models",
    )
    isolated_paths = get_paths(tmp_path / "isolated_app")

    storage = import_module("src.00_common.storage")
    feature_storage = import_module("src.30_ai_modeling.feature_engineering.storage")
    freshness = import_module("src.30_ai_modeling.feature_engineering.freshness")
    top3_model = import_module("src.30_ai_modeling.tasks.top3.model")
    monkeypatch.setattr(storage, "PATHS", isolated_paths)
    monkeypatch.setattr(feature_storage, "PATHS", isolated_paths)
    monkeypatch.setattr(freshness, "PATHS", isolated_paths)
    monkeypatch.setattr(top3_model, "PATHS", isolated_paths)
    monkeypatch.setattr(requests.sessions.Session, "request", _offline_request)

    parsed = _parse_one_saved_result_html(isolated_paths.logs)
    save_race_records(
        _minimum_training_history(parsed),
        run_id="offline-e2e",
        dataset_type="historical",
    )
    history = load_records("historical")
    assert len(history) >= 500

    feature_db = isolated_paths.feature_database
    assert generate_recent_form_features(
        RecentFormRunConfig(), feature_database=feature_db
    )["status"] == "completed"
    assert generate_speed_index_features(
        SpeedIndexRunConfig(), feature_database=feature_db
    )["status"] == "completed"
    assert generate_recent_speed_features(
        RecentSpeedRunConfig(), feature_database=feature_db
    )["status"] == "completed"
    assert generate_race_entry_features(
        RaceEntryRunConfig(), feature_database=feature_db
    )["status"] == "completed"

    metrics = train_model(
        history,
        TrainConfig(
            epochs=1,
            batch_size=128,
            hidden_dim=8,
            dropout=0.0,
            patience=1,
            random_seed=42,
        ),
        log=lambda _: None,
    )
    model_dir = isolated_paths.models / "top3" / str(metrics["model_run_id"])
    target_race_id = str(history.iloc[-1]["race_id"])
    comparison, output_path, summary = predict_historical_race(
        history,
        target_race_id,
        model_dir,
    )

    assert output_path.is_relative_to(isolated_paths.predictions)
    assert output_path.exists()
    assert len(comparison) == len(parsed)
    assert comparison["top3_probability"].between(0, 1).all()
    assert comparison["finish_position"].notna().all()
    assert {"predicted_top3", "actual_top3", "top3_hit"}.issubset(comparison)
    assert summary["race_id"] == target_race_id
    assert isolated_paths.database.exists()
    assert isolated_paths.feature_database.exists()
    assert model_dir.exists()

    prediction_date = date(2025, 3, 1)
    upcoming_races = []
    for race_number in (1, 2):
        upcoming = parsed.copy()
        upcoming["race_id"] = f"upcoming_{race_number}"
        upcoming["race_date"] = prediction_date
        upcoming["race_number"] = race_number
        upcoming["race_name"] = f"今後レース{race_number}"
        upcoming["dataset_type"] = "upcoming"
        upcoming["finish_position"] = pd.NA
        upcoming_races.append(upcoming)
    upcoming = pd.concat(upcoming_races, ignore_index=True)
    date_predictions, date_output_path = predict_race_date(
        history, upcoming, prediction_date, model_dir
    )
    assert date_predictions["race_id"].nunique() == 2
    assert (date_predictions.groupby("race_id")["prediction_rank"].min() == 1).all()
    assert date_output_path.is_relative_to(isolated_paths.predictions)

    static_page = build_prediction_site(
        date_predictions,
        prediction_date,
        str(metrics["model_run_id"]),
        tmp_path / "pages",
    )
    assert static_page.exists()
    assert (tmp_path / "pages" / "index.html").exists()
    page_html = static_page.read_text(encoding="utf-8")
    assert "今後レース1" in page_html
    assert "3着以内確率" in page_html

    protected_after = _file_fingerprints(
        PROJECT_ROOT / "data" / "racing.duckdb",
        PROJECT_ROOT / "data" / "features.duckdb",
        PROJECT_ROOT / "models",
    )
    assert protected_after == protected_before
