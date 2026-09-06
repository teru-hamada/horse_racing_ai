from __future__ import annotations

import pandas as pd

from src.public_api import (
    build_prediction_site,
    latest_prediction_file,
    prediction_date_status,
)


def test_static_prediction_site_escapes_content_and_updates_index(tmp_path):
    predictions = pd.DataFrame([
        {
            "race_id": "r1",
            "race_number": 1,
            "race_name": "<script>alert(1)</script>",
            "course_name": "札幌",
            "horse_number": 3,
            "horse_name": "テストホース",
            "jockey_name": "テスト騎手",
            "prediction_rank": 1,
            "top3_probability": 0.75,
            "odds": 4.2,
            "expected_value_index": 3.15,
        }
    ])

    page = build_prediction_site(
        predictions, "2026-08-29", "model-test", tmp_path / "docs"
    )

    html = page.read_text(encoding="utf-8")
    index = (tmp_path / "docs" / "index.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "75.0%" in html
    assert '<details class="race">' in html
    assert (
        "<summary>札幌 1R &lt;script&gt;alert(1)&lt;/script&gt;</summary>"
        in html
    )
    assert "出走馬数" not in html
    assert "model-test" not in html
    assert "r1" not in html
    assert "predictions/2026-08-29.html" in index


def test_graded_race_title_gets_gold_style_class(tmp_path):
    predictions = pd.DataFrame([{
        "race_id": "graded",
        "race_number": 11,
        "race_name": "レパードステークス(G3)",
        "course_name": "新潟",
        "horse_number": 1,
        "horse_name": "テストホース",
        "jockey_name": "テスト騎手",
        "prediction_rank": 1,
        "top3_probability": 0.6,
        "odds": 3.0,
        "expected_value_index": 1.8,
    }])

    page = build_prediction_site(
        predictions, "2026-08-30", "model-test", tmp_path / "docs"
    )
    html = page.read_text(encoding="utf-8")

    assert '<details class="race graded">' in html
    assert ".race.graded>summary{color:var(--gold)}" in html


def test_static_prediction_site_includes_result_comparison(tmp_path):
    predictions = pd.DataFrame([
        {
            "race_id": "race-1",
            "race_number": 1,
            "race_name": "未勝利",
            "course_name": "中京",
            "horse_id": "horse-1",
            "horse_number": 4,
            "horse_name": "テストホース",
            "jockey_name": "テスト騎手",
            "prediction_rank": 1,
            "top3_probability": 0.7,
            "odds": 2.5,
            "expected_value_index": 1.75,
        }
    ])
    comparison = predictions.assign(
        finish_position=2,
        predicted_top3=True,
        actual_top3=True,
        top3_hit=True,
    )
    summary = {
        "requested_races": 2,
        "compared_races": 1,
        "failed_races": 1,
        "top3_hit_count": 1,
        "perfect_top3_races": 0,
        "failures": [{"race_id": "race-2", "message": "結果未確定"}],
    }

    page = build_prediction_site(
        predictions,
        "2026-09-06",
        "model-test",
        tmp_path / "docs",
        comparison=comparison,
        comparison_summary=summary,
    )
    html = page.read_text(encoding="utf-8")

    assert "確定結果との比較" in html
    assert "比較完了: 1 / 2 レース" in html
    assert "予測上位3頭の的中: 1 頭" in html
    assert "実着順" in html
    assert '<td>2</td><td class="hit">的中</td>' in html
    assert "race-2 (結果未確定)" in html


def test_prediction_date_status_survives_application_restart(tmp_path):
    predictions_dir = tmp_path / "data" / "predictions"
    site_dir = tmp_path / "docs"

    assert prediction_date_status(
        "2026-09-06", predictions_dir, site_dir
    ) == (False, False)

    run_dir = predictions_dir / "date_pred_test"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions_2026-09-06.parquet").write_bytes(b"saved")
    assert latest_prediction_file("2026-09-06", predictions_dir) == (
        run_dir / "predictions_2026-09-06.parquet"
    )
    assert prediction_date_status(
        "2026-09-06", predictions_dir, site_dir
    ) == (True, False)

    comparison_dir = site_dir / "predictions"
    comparison_dir.mkdir(parents=True)
    (comparison_dir / "2026-09-06.html").write_text(
        '<section data-comparison-status="completed"></section>',
        encoding="utf-8",
    )
    assert prediction_date_status(
        "2026-09-06", predictions_dir, site_dir
    ) == (True, True)
    assert latest_prediction_file("invalid", predictions_dir) is None
