from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd


COURSES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
SURFACES = ["芝", "ダート"]
DISTANCES = [1200, 1400, 1600, 1800, 2000, 2200, 2400]


def generate_demo_records(seed: int = 42, historical_races: int = 360, upcoming_races: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    horse_count = 500
    jockey_count = 45
    horse_skill = rng.normal(0, 1, horse_count)
    jockey_skill = rng.normal(0, 0.35, jockey_count)
    horse_ids = np.array([f"H{i:05d}" for i in range(horse_count)])
    horse_names = np.array([f"デモホース{i:03d}" for i in range(horse_count)])
    sexes = rng.choice(["牡", "牝", "セ"], size=horse_count, p=[0.48, 0.43, 0.09])
    ages = rng.integers(3, 8, size=horse_count)

    start_date = date.today() - timedelta(days=historical_races * 3)
    rows: list[dict[str, object]] = []
    for race_index in range(historical_races):
        race_date = start_date + timedelta(days=race_index * 3)
        field_size = int(rng.integers(10, 17))
        selected = rng.choice(horse_count, size=field_size, replace=False)
        jockeys = rng.choice(jockey_count, size=field_size, replace=False)
        course = COURSES[race_index % len(COURSES)]
        surface = SURFACES[race_index % 2]
        distance = DISTANCES[race_index % len(DISTANCES)]
        carried = rng.choice([51, 52, 53, 54, 55, 56, 57, 58], size=field_size)
        body = rng.normal(480, 28, field_size).round()
        body_change = rng.integers(-12, 13, field_size)
        raw_score = horse_skill[selected] + jockey_skill[jockeys] - 0.02 * np.abs(carried - 55) + rng.normal(0, 0.75, field_size)
        finish_order = np.argsort(-raw_score)
        finish_position = np.empty(field_size, dtype=int)
        finish_position[finish_order] = np.arange(1, field_size + 1)
        implied = np.exp(raw_score - raw_score.max())
        probs = implied / implied.sum()
        odds = np.maximum(1.1, (1 / np.clip(probs, 0.01, 0.85)) * rng.uniform(0.72, 1.05, field_size)).round(1)
        popularity_order = np.argsort(odds)
        popularity = np.empty(field_size, dtype=int)
        popularity[popularity_order] = np.arange(1, field_size + 1)
        race_id = f"D{race_date:%Y%m%d}{race_index % 99:02d}"
        for i, horse_index in enumerate(selected):
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "course_name": course,
                    "race_number": race_index % 12 + 1,
                    "race_name": f"デモ過去レース{race_index + 1}",
                    "surface": surface,
                    "distance": distance,
                    "weather": rng.choice(["晴", "曇", "雨"]),
                    "track_condition": rng.choice(["良", "稍重", "重"], p=[0.7, 0.2, 0.1]),
                    "horse_id": horse_ids[horse_index],
                    "horse_name": horse_names[horse_index],
                    "horse_number": i + 1,
                    "frame_number": min(8, (i // 2) + 1),
                    "sex": sexes[horse_index],
                    "age": int(ages[horse_index]),
                    "carried_weight": float(carried[i]),
                    "jockey_id": f"J{jockeys[i]:03d}",
                    "jockey_name": f"デモ騎手{jockeys[i]:02d}",
                    "trainer_id": f"T{horse_index % 80:03d}",
                    "trainer_name": f"デモ調教師{horse_index % 80:02d}",
                    "odds": float(odds[i]),
                    "popularity": float(popularity[i]),
                    "body_weight": float(body[i]),
                    "body_weight_change": float(body_change[i]),
                    "finish_position": float(finish_position[i]),
                    "time_seconds": float(distance / 16.5 + rng.normal(0, 1.2)),
                    "dataset_type": "historical",
                }
            )

    historical = pd.DataFrame(rows)
    upcoming_rows: list[dict[str, object]] = []
    future_date = date.today() + timedelta(days=(5 - date.today().weekday()) % 7)
    for race_index in range(upcoming_races):
        field_size = int(rng.integers(12, 17))
        selected = rng.choice(horse_count, size=field_size, replace=False)
        jockeys = rng.choice(jockey_count, size=field_size, replace=False)
        raw_score = horse_skill[selected] + jockey_skill[jockeys] + rng.normal(0, 0.5, field_size)
        implied = np.exp(raw_score - raw_score.max())
        probs = implied / implied.sum()
        odds = np.maximum(1.1, (1 / np.clip(probs, 0.01, 0.85)) * rng.uniform(0.75, 1.08, field_size)).round(1)
        popularity_order = np.argsort(odds)
        popularity = np.empty(field_size, dtype=int)
        popularity[popularity_order] = np.arange(1, field_size + 1)
        race_id = f"U{future_date:%Y%m%d}{race_index + 1:02d}"
        for i, horse_index in enumerate(selected):
            upcoming_rows.append(
                {
                    "race_id": race_id,
                    "race_date": future_date,
                    "course_name": COURSES[race_index % len(COURSES)],
                    "race_number": race_index + 1,
                    "race_name": f"デモ週末レース{race_index + 1}",
                    "surface": SURFACES[race_index % 2],
                    "distance": DISTANCES[race_index % len(DISTANCES)],
                    "weather": None,
                    "track_condition": None,
                    "horse_id": horse_ids[horse_index],
                    "horse_name": horse_names[horse_index],
                    "horse_number": i + 1,
                    "frame_number": min(8, (i // 2) + 1),
                    "sex": sexes[horse_index],
                    "age": int(ages[horse_index]),
                    "carried_weight": float(rng.choice([52, 53, 54, 55, 56, 57, 58])),
                    "jockey_id": f"J{jockeys[i]:03d}",
                    "jockey_name": f"デモ騎手{jockeys[i]:02d}",
                    "trainer_id": f"T{horse_index % 80:03d}",
                    "trainer_name": f"デモ調教師{horse_index % 80:02d}",
                    "odds": float(odds[i]),
                    "popularity": float(popularity[i]),
                    "body_weight": float(rng.normal(480, 28)),
                    "body_weight_change": float(rng.integers(-10, 11)),
                    "finish_position": None,
                    "time_seconds": None,
                    "dataset_type": "upcoming",
                }
            )
    return historical, pd.DataFrame(upcoming_rows)
