"""Register and validate a single race in a new, explicitly isolated database."""
from __future__ import annotations

from datetime import date
from importlib import import_module
from pathlib import Path

import pandas as pd

storage = import_module("src.00_common.storage")


def _counts(odds: pd.DataFrame) -> dict:
    return odds.groupby("bet_type").size().to_dict()


def _same_values(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    # Compare payloads independent of row order, dtype, and registration timestamps.
    def canonical(frame):
        return frame.reindex(columns=columns).sort_values(columns, na_position="last").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(canonical(left), canonical(right), check_dtype=False)
        return True
    except AssertionError:
        return False


def verify_database(card: pd.DataFrame, odds: pd.DataFrame, race_date: date,
                    race_id: str, output: Path) -> dict:
    from .html_acquisition import inspect_card

    database = output / "smoke.duckdb"
    if database.resolve() == storage.PATHS.database.resolve() or database.exists():
        raise ValueError("検証DBには新しい専用ファイルが必要です。")
    checks = {}
    expected_counts = _counts(odds)
    snapshots = []
    for attempt in (1, 2):
        run_id = f"database-smoke-{attempt}"
        if not card.empty:
            storage.save_race_records(card, run_id, "upcoming", database=database)
        storage.save_race_odds(odds, run_id, database=database)
        loaded_card = storage.load_records("upcoming", database=database)
        # Exercise the same race-ID based lookup used by predictions.
        loaded_odds = storage.load_race_odds([race_id], database=database)
        with storage.connect(database) as con:
            total_odds = con.execute("SELECT count(*) FROM race_odds").fetchone()[0]
        snapshots.append((loaded_card, loaded_odds))
        prefix = f"registration_{attempt}"
        checks[f"{prefix}_runner_count"] = len(loaded_card) == len(card) > 0
        checks[f"{prefix}_odds_counts"] = (
            len(loaded_odds) == len(odds) == total_odds and len(odds) > 0
            and _counts(loaded_odds) == expected_counts
        )
        checks[f"{prefix}_horse_and_frame_numbers"] = inspect_card(loaded_card)["status"] == "ok"
        checks[f"{prefix}_race_and_date"] = all(
            not frame.empty
            and frame["race_id"].astype(str).eq(race_id).all()
            and pd.to_datetime(frame["race_date"]).dt.date.eq(race_date).all()
            for frame in (loaded_card, loaded_odds)
        )

    first_card, first_odds = snapshots[0]
    card_columns = [c for c in storage.RACE_RECORD_COLUMNS if c not in {"collection_run_id", "collected_at"}]
    odds_columns = [c for c in storage.ODDS_COLUMNS if c not in {"collection_run_id", "registered_at"}]
    checks["reregistration_unchanged"] = (
        _same_values(first_card, loaded_card, card_columns)
        and _same_values(first_odds, loaded_odds, odds_columns)
    )
    checks["reregistration_replaced"] = all(
        not frame.empty and frame["collection_run_id"].eq("database-smoke-2").all()
        for frame in (loaded_card, loaded_odds)
    )
    checks["unique_odds_selections"] = not loaded_odds.duplicated(
        ["race_id", "bet_type", "selection_1", "selection_2", "selection_3"]
    ).any()
    win = loaded_odds[loaded_odds["bet_type"].eq("win")]
    joined = loaded_card[["race_id", "horse_number", "horse_name"]].merge(
        win[["race_id", "selection_1", "odds_min"]],
        left_on=["race_id", "horse_number"], right_on=["race_id", "selection_1"],
        how="outer", indicator=True,
    )
    matched = joined["_merge"].eq("both") & joined["odds_min"].gt(0)
    checks["win_odds_match_all_runners"] = (
        len(joined) == len(loaded_card) > 0 and bool(matched.all())
        and not win.duplicated(["race_id", "selection_1"]).any()
    )
    loaded_card.to_csv(output / "db_card.csv", index=False, encoding="utf-8-sig")
    loaded_odds.to_csv(output / "db_odds.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(output / "db_win_join.csv", index=False, encoding="utf-8-sig")
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "ok" if all(checks.values()) else "incomplete",
        "database": database.name, "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "expected_runners": len(card), "registered_runners": len(loaded_card),
        "expected_odds_by_bet_type": expected_counts,
        "registered_odds_by_bet_type": _counts(loaded_odds),
        "matched_win_runners": int(matched.sum()),
        "unmatched_card_numbers": joined.loc[joined["_merge"].eq("left_only"), "horse_number"].dropna().astype(int).tolist(),
        "unmatched_odds_numbers": joined.loc[joined["_merge"].eq("right_only"), "selection_1"].dropna().astype(int).tolist(),
    }
