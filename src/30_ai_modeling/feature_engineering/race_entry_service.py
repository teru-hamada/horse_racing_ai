from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Callable

from .context import FeatureContext
from .freshness import source_data_state, source_state_token
from .generators import RaceEntryGenerator, prepare_race_entries
from .storage import replace_features, save_feature_run


_source_storage = import_module("src.00_common.storage")


@dataclass(frozen=True)
class RaceEntryRunConfig:
    feature_set_name: str = "race_entry"
    feature_set_version: str = "1.1.0"


class RaceEntryGenerationCancelled(Exception):
    """Raised when generation is cancelled before persistence."""


def generate_race_entry_features(
    config: RaceEntryRunConfig = RaceEntryRunConfig(),
    *,
    feature_database: Path | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    run_id = f"race_entry_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    generated_at = datetime.now()

    def check_cancelled() -> None:
        if is_cancelled and is_cancelled():
            raise RaceEntryGenerationCancelled("Race-entry generation was cancelled")

    metadata = {
        "feature_run_id": run_id,
        "feature_set_name": config.feature_set_name,
        "feature_set_version": config.feature_set_version,
        "generated_at": generated_at,
        "row_count": 0,
        "config": asdict(config),
        "status": "running",
    }
    save_feature_run(metadata, feature_database)
    try:
        check_cancelled()
        initial_state = source_data_state()
        if log:
            log("Loading historical race-entry data")
        raw = _source_storage.load_records("historical")
        entries = prepare_race_entries(raw)
        targets = entries[["race_id", "horse_id", "race_date"]].copy()
        generated = RaceEntryGenerator(
            progress=progress, check_cancelled=check_cancelled
        ).transform(targets, entries, FeatureContext(
            config.feature_set_name, config.feature_set_version, {}
        ))
        generated["race_date"] = targets["race_date"].dt.date
        generated["feature_set_name"] = config.feature_set_name
        generated["feature_set_version"] = config.feature_set_version
        generated["history_cutoff"] = targets["race_date"]
        generated["feature_run_id"] = run_id
        generated["generated_at"] = generated_at
        final_state = source_data_state()
        check_cancelled()
        if final_state != initial_state:
            raise RuntimeError("Historical data changed during race-entry generation")
        if log:
            log("Saving race-entry features")
        replace_features(generated, feature_database)
        completed = {
            **metadata,
            "status": "completed",
            "row_count": len(generated),
            "start_date": targets["race_date"].min().date(),
            "end_date": targets["race_date"].max().date(),
            "source_data_fingerprint": source_state_token(final_state),
            "source_state": final_state,
            "source_state_token": source_state_token(final_state),
            "message": "Race-entry feature generation completed",
        }
        save_feature_run(completed, feature_database)
        return completed
    except Exception as exc:
        status = "cancelled" if isinstance(exc, RaceEntryGenerationCancelled) else "failed"
        save_feature_run({**metadata, "status": status, "message": str(exc)}, feature_database)
        raise
