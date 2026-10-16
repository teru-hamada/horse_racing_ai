# 検証対象：履歴特徴量が当該行の結果を使わないことと、未開催レースへの特徴量生成を確認する。
# 血統カテゴリの採用、馬体重を入力特徴量へ含めない方針も検証する。
#
from src.public_api import (
    generate_demo_records,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_prediction_frame,
    build_training_frame,
)


# 馬ごとの初出走時点のprior_startsがゼロとなり、当該走を過去成績へ数えない。
def test_training_features_do_not_use_current_result():
    historical, upcoming = generate_demo_records(historical_races=30, upcoming_races=2)
    training = build_training_frame(historical)
    assert "target_top3" in training.columns
    assert training["prior_starts"].min() == 0
    first_horse_rows = training.sort_values("race_date").groupby("horse_id").head(1)
    assert (first_horse_rows["prior_starts"] == 0).all()


# 未開催の全出走馬へ履歴特徴量を付け、着順は欠損のまま保持する。
def test_prediction_features_are_created():
    historical, upcoming = generate_demo_records(historical_races=30, upcoming_races=2)
    prediction = build_prediction_frame(historical, upcoming)
    assert len(prediction) == len(upcoming)
    assert prediction["finish_position"].isna().all()
    assert prediction["prior_top3_rate"].notna().all()


# 父・母・母父のIDがカテゴリ特徴量として宣言されていることを確認する。
def test_pedigree_is_used_as_categorical_features():
    assert "sire_id" in CATEGORICAL_FEATURES
    assert "dam_id" in CATEGORICAL_FEATURES
    assert "damsire_id" in CATEGORICAL_FEATURES


# 馬体重と増減を数値特徴量の定義へ含めない方針を固定する。
def test_body_weight_is_not_used_as_model_feature():
    assert "body_weight" not in NUMERIC_FEATURES
    assert "body_weight_change" not in NUMERIC_FEATURES
