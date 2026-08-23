from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from importlib import import_module
from pathlib import Path
from typing import Callable

import pandas as pd

from .generators import RecentFormGenerator, prepare_historical_performances
from .freshness import source_data_state, source_state_token
from .pipeline import FeaturePipeline
from .registry import FeatureRegistry, FeatureSetDefinition
from .storage import replace_features, save_feature_run


_source_storage = import_module("src.00_common.storage")


@dataclass(frozen=True)
class RecentFormRunConfig:
    feature_set_name: str = "baseline"
    feature_set_version: str = "1.1.0"
    half_life_days: float = 180.0
    max_lookback_days: int = 1095
    start_date: date | None = None
    end_date: date | None = None


class FeatureGenerationCancelled(Exception):
    """Raised when a user requests cancellation before feature persistence."""


def _source_fingerprint(history: pd.DataFrame) -> str:
    columns = [
        "race_id", "horse_id", "race_date", "finish_position", "last_3f_time"
    ]
    ordered = history.loc[:, columns].sort_values(
        ["race_date", "race_id", "horse_id"], kind="stable"
    )
    hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update(hashes.tobytes())
    digest.update(str(len(ordered)).encode("ascii"))
    return digest.hexdigest()


def generate_recent_form_features(
    config: RecentFormRunConfig = RecentFormRunConfig(),
    *,
    feature_database: Path | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Generate and atomically persist recent-form features from source history."""

    run_id = f"features_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    generated_at = datetime.now()
    def check_cancelled() -> None:
        if is_cancelled and is_cancelled():
            raise FeatureGenerationCancelled("Feature generation was cancelled")

    generator = RecentFormGenerator(
        progress=(
            (lambda value: progress(0.15 + 0.75 * value))
            if progress else None
        ),
        check_cancelled=check_cancelled,
    )
    run_config = {
        **asdict(config),
        "generators": {generator.name: generator.version},
    }
    running_metadata = {
        "feature_run_id": run_id,
        "feature_set_name": config.feature_set_name,
        "feature_set_version": config.feature_set_version,
        "generated_at": generated_at,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "row_count": 0,
        "config": run_config,
        "status": "running",
    }
    save_feature_run(running_metadata, feature_database)

    try:
        if progress:
            progress(0.02)
        check_cancelled()
        if log:
            log("Loading and normalizing historical race results")
        initial_source_state = source_data_state()
        raw_history = _source_storage.load_records("historical")
        history = prepare_historical_performances(raw_history)
        if progress:
            progress(0.15)
        check_cancelled()
        targets = history[["race_id", "horse_id", "race_date"]].copy()
        if config.start_date is not None:
            targets = targets[targets["race_date"].dt.date >= config.start_date]
        if config.end_date is not None:
            targets = targets[targets["race_date"].dt.date <= config.end_date]
        targets = targets.reset_index(drop=True)
        if targets.empty:
            raise ValueError("No historical races exist in the requested target period")

        definition = FeatureSetDefinition(
            name=config.feature_set_name,
            version=config.feature_set_version,
            generators=(generator.name,),
            parameters={
                "recent_form": {
                    "half_life_days": config.half_life_days,
                    "max_lookback_days": config.max_lookback_days,
                }
            },
        )
        if log:
            log(f"Generating recent-form features for {len(targets):,} runners")
        pipeline = FeaturePipeline(FeatureRegistry([generator]))
        features = pipeline.transform(targets, history, definition)
        features["race_date"] = targets["race_date"].dt.date
        features["feature_set_name"] = config.feature_set_name
        features["feature_set_version"] = config.feature_set_version
        features["history_cutoff"] = targets["race_date"]
        features["feature_run_id"] = run_id
        features["generated_at"] = generated_at

        final_source_state = source_data_state()
        if final_source_state != initial_source_state:
            raise RuntimeError(
                "Historical source data changed during feature generation; run it again"
            )

        if log:
            log("Replacing the stored feature-set version atomically")
        if progress:
            progress(0.92)
        check_cancelled()
        replace_features(features, feature_database)
        fingerprint = _source_fingerprint(history)
        completed = {
            **running_metadata,
            "source_data_fingerprint": fingerprint,
            "source_state_token": source_state_token(final_source_state),
            "source_state": final_source_state,
            "start_date": targets["race_date"].min().date(),
            "end_date": targets["race_date"].max().date(),
            "row_count": len(features),
            "status": "completed",
            "message": "Recent-form feature generation completed",
        }
        save_feature_run(completed, feature_database)
        if progress:
            progress(1.0)
        if log:
            log(f"Saved {len(features):,} feature rows")
        return completed
    except Exception as exc:
        status = "cancelled" if isinstance(exc, FeatureGenerationCancelled) else "failed"
        save_feature_run(
            {
                **running_metadata,
                "status": status,
                "message": str(exc),
            },
            feature_database,
        )
        raise
