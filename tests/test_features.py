from src.demo_data import generate_demo_records
from src.features import build_prediction_frame, build_training_frame


def test_training_features_do_not_use_current_result():
    historical, upcoming = generate_demo_records(historical_races=30, upcoming_races=2)
    training = build_training_frame(historical)
    assert "target_top3" in training.columns
    assert training["prior_starts"].min() == 0
    first_horse_rows = training.sort_values("race_date").groupby("horse_id").head(1)
    assert (first_horse_rows["prior_starts"] == 0).all()


def test_prediction_features_are_created():
    historical, upcoming = generate_demo_records(historical_races=30, upcoming_races=2)
    prediction = build_prediction_frame(historical, upcoming)
    assert len(prediction) == len(upcoming)
    assert prediction["finish_position"].isna().all()
    assert prediction["prior_top3_rate"].notna().all()
