from src.demo_data import generate_demo_records
from src.storage import normalize_race_frame


def test_normalize_race_frame_has_expected_columns():
    historical, _ = generate_demo_records(historical_races=5, upcoming_races=1)
    historical["collection_run_id"] = "test"
    historical["collected_at"] = "2026-01-01"
    normalized = normalize_race_frame(historical)
    assert "race_id" in normalized.columns
    assert "finish_position" in normalized.columns
    assert len(normalized) == len(historical)
