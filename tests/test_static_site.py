from __future__ import annotations

import pandas as pd

from src.public_api import build_prediction_site


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
