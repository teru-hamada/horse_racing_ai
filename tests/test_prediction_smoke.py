import json
from dataclasses import replace
from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src import prediction_bundle as bundle
from src.prediction_smoke import compare_results, validate_predictions


@pytest.fixture
def source(tmp_path, monkeypatch):
    paths = replace(bundle.PATHS, database=tmp_path / "source.duckdb",
                    feature_database=tmp_path / "features.duckdb", models=tmp_path / "models")
    monkeypatch.setattr(bundle, "PATHS", paths)
    model = paths.models / "top3/model_new"
    model.mkdir(parents=True)
    for name in bundle.MODEL_FILES:
        (model / name).write_text(json.dumps({"model_run_id": "model_new"}), encoding="utf-8")
    with duckdb.connect(str(paths.database)) as con:
        con.execute("CREATE TABLE model_runs(model_run_id VARCHAR,created_at TIMESTAMP,model_path VARCHAR,status VARCHAR,task_name VARCHAR)")
        con.executemany("INSERT INTO model_runs VALUES (?,?,?,?,?)", [
            ("model_old", datetime(2026, 1, 1), "missing", "completed", "top3"),
            ("model_new", datetime(2026, 9, 1), str(model), "completed", "top3"),
            ("model_failed", datetime(2026, 9, 2), "missing", "failed", "top3"),
        ])
        con.execute("CREATE TABLE race_records(dataset_type VARCHAR)")
        con.execute("INSERT INTO race_records VALUES ('historical'),('upcoming')")
    with duckdb.connect(str(paths.feature_database)) as con:
        con.execute("CREATE TABLE performance_features(performance_feature_name VARCHAR,performance_feature_version VARCHAR)")
        con.execute("INSERT INTO performance_features VALUES ('speed_index','1.0.0')")
    return paths, model


def test_export_latest_and_read_only_source(tmp_path, source):
    paths, _ = source
    before = [bundle.sha256(p) for p in (paths.database, paths.feature_database)]
    manifest = bundle.export_bundle(tmp_path / "bundle")
    assert manifest["model_id"] == "model_new"
    unpacked = bundle.unpack_bundle(tmp_path / "bundle", tmp_path / "work")
    assert unpacked == manifest
    assert before == [bundle.sha256(p) for p in (paths.database, paths.feature_database)]
    with duckdb.connect(str(tmp_path / "work/racing.duckdb"), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM race_records").fetchone()[0] == 1


def test_missing_latest_file_does_not_fall_back(tmp_path, source):
    paths, model = source
    (model / "model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="最新モデル"):
        bundle.export_bundle(tmp_path / "bundle")
    assert not (tmp_path / "bundle/runtime.zip").exists()


def test_tampering_rejected_before_deserialization(tmp_path, source):
    bundle.export_bundle(tmp_path / "bundle")
    with (tmp_path / "bundle/runtime.zip").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.unpack_bundle(tmp_path / "bundle", tmp_path / "work")


@pytest.mark.parametrize("probability", [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_probability_rejected(probability):
    card = pd.DataFrame({"race_id": ["r"], "horse_id": ["h"], "horse_number": [1]})
    with pytest.raises(ValueError, match="予測確率"):
        validate_predictions(card, card.assign(top3_probability=probability))


def test_missing_or_wrong_runner_rejected():
    card = pd.DataFrame({"race_id": ["r"], "horse_id": ["h"], "horse_number": [1]})
    with pytest.raises(ValueError):
        validate_predictions(card, card.iloc[:0].assign(top3_probability=0.5))
    with pytest.raises(ValueError):
        validate_predictions(card, card.assign(horse_id="other", top3_probability=0.5))
    validate_predictions(card, card.assign(top3_probability=0.5))


def test_replay_comparison_detects_changes(tmp_path):
    reference, output = tmp_path / "reference", tmp_path / "output"
    for folder in (reference, output):
        folder.mkdir()
        pd.DataFrame({"race_id": ["r"], "horse_id": ["h"], "horse_number": [1],
                      "top3_probability": [0.5]}).to_csv(folder / "predictions.csv", index=False)
        pd.DataFrame(columns=["race_id", "bet_type", "selection"]).to_csv(folder / "bets.csv", index=False)
    assert compare_results(output, reference)["status"] == "ok"
    predictions = pd.read_csv(output / "predictions.csv")
    predictions["top3_probability"] = 0.6
    predictions.to_csv(output / "predictions.csv", index=False)
    assert compare_results(output, reference)["status"] == "error"
