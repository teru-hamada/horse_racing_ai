from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Callable

from .context import FeatureContext
from .freshness import performance_feature_freshness, source_data_state, source_state_token
from .generators import RecentSpeedGenerator, prepare_historical_performances
from .storage import (
    load_performance_features,
    performance_feature_summary,
    replace_features,
    save_feature_run,
)


_source_storage = import_module("src.00_common.storage")


@dataclass(frozen=True)
class RecentSpeedRunConfig:
    feature_set_name: str = "recent_speed"
    feature_set_version: str = "1.0.0"
    dependency_name: str = "speed_index"
    dependency_version: str = "1.0.0"
    half_life_days: float = 180.0
    max_lookback_days: int = 1095


class RecentSpeedGenerationCancelled(Exception):
    """Raised when generation is cancelled before persistence."""


def generate_recent_speed_features(
    config: RecentSpeedRunConfig = RecentSpeedRunConfig(),
    *,
    feature_database: Path | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    run_id = f"recent_speed_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    generated_at = datetime.now()

    def check_cancelled() -> None:
        if is_cancelled and is_cancelled():
            raise RecentSpeedGenerationCancelled("Recent-speed generation was cancelled")

    dependency_freshness = performance_feature_freshness(
        config.dependency_name,
        config.dependency_version,
        feature_database=feature_database,
    )
    dependency_summary = performance_feature_summary(
        config.dependency_name, config.dependency_version, feature_database
    )
    if dependency_freshness["status"] != "fresh":
        raise RuntimeError("The dependent speed index is missing or stale; generate it first")
    dependency_run_id = dependency_summary["latest_run_id"]
    run_config = {
        **asdict(config),
        "dependency_run_id": dependency_run_id,
        "dependency_fingerprint": dependency_summary["source_data_fingerprint"],
    }
    metadata = {
        "feature_run_id": run_id,
        "feature_set_name": config.feature_set_name,
        "feature_set_version": config.feature_set_version,
        "generated_at": generated_at,
        "row_count": 0,
        "config": run_config,
        "status": "running",
    }
    save_feature_run(metadata, feature_database)
    try:
        check_cancelled()
        initial_state = source_data_state()
        if log:
            log("Loading targets and one-run speed indexes")
        raw = _source_storage.load_records("historical")
        prepared = prepare_historical_performances(raw)
        targets = prepared[["race_id", "horse_id", "race_date"]].copy()
        speed_history = load_performance_features(
            config.dependency_name, config.dependency_version, feature_database
        )
        generated = RecentSpeedGenerator(
            progress=progress,
            check_cancelled=check_cancelled,
        ).transform(
            targets,
            speed_history,
            FeatureContext(
                config.feature_set_name,
                config.feature_set_version,
                {"recent_speed": {
                    "half_life_days": config.half_life_days,
                    "max_lookback_days": config.max_lookback_days,
                }},
            ),
        )
        generated["race_date"] = targets["race_date"].dt.date
        generated["feature_set_name"] = config.feature_set_name
        generated["feature_set_version"] = config.feature_set_version
        generated["history_cutoff"] = targets["race_date"]
        generated["feature_run_id"] = run_id
        generated["generated_at"] = generated_at
        final_state = source_data_state()
        check_cancelled()
        if final_state != initial_state:
            raise RuntimeError("Historical data changed during recent-speed generation")
        current_dependency = performance_feature_summary(
            config.dependency_name, config.dependency_version, feature_database
        )
        if current_dependency["latest_run_id"] != dependency_run_id:
            raise RuntimeError("The speed-index dependency changed during generation")
        if log:
            log("Saving recent-speed aggregates")
        replace_features(generated, feature_database)
        digest = hashlib.sha256()
        digest.update(str(dependency_run_id).encode("utf-8"))
        digest.update(source_state_token(final_state).encode("ascii"))
        completed = {
            **metadata,
            "status": "completed",
            "row_count": len(generated),
            "start_date": targets["race_date"].min().date(),
            "end_date": targets["race_date"].max().date(),
            "source_data_fingerprint": digest.hexdigest(),
            "source_state": final_state,
            "source_state_token": source_state_token(final_state),
            "message": "Recent-speed feature generation completed",
        }
        save_feature_run(completed, feature_database)
        return completed
    except Exception as exc:
        status = "cancelled" if isinstance(exc, RecentSpeedGenerationCancelled) else "failed"
        save_feature_run({**metadata, "status": status, "message": str(exc)}, feature_database)
        raise
