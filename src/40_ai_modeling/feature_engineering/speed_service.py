from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Callable

import pandas as pd

from .context import FeatureContext
from .freshness import source_data_state, source_state_token
from .generators import SpeedIndexGenerator, prepare_speed_performances
from .storage import replace_performance_features, save_feature_run


_source_storage = import_module("src.00_common.storage")


@dataclass(frozen=True)
class SpeedIndexRunConfig:
    feature_name: str = "speed_index"
    feature_version: str = "1.0.0"
    half_life_days: float = 730.0
    max_lookback_days: int = 3650
    seconds_scale_at_1600m: float = 10.0
    weight_points_per_kg: float = 1.5


class SpeedIndexGenerationCancelled(Exception):
    """Raised when speed-index generation is cancelled before persistence."""


def _fingerprint(frame: pd.DataFrame) -> str:
    columns = list(prepare_speed_performances(frame).columns)
    ordered = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    ordered = ordered.sort_values(["race_date", "race_id", "horse_id"], kind="stable")
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
    digest.update(str(len(ordered)).encode("ascii"))
    return digest.hexdigest()


def generate_speed_index_features(
    config: SpeedIndexRunConfig = SpeedIndexRunConfig(),
    *,
    feature_database: Path | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    run_id = f"speed_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    generated_at = datetime.now()
    metadata = {
        "feature_run_id": run_id,
        "feature_set_name": config.feature_name,
        "feature_set_version": config.feature_version,
        "generated_at": generated_at,
        "row_count": 0,
        "config": asdict(config),
        "status": "running",
    }
    save_feature_run(metadata, feature_database)
    try:
        def check_cancelled() -> None:
            if is_cancelled and is_cancelled():
                raise SpeedIndexGenerationCancelled(
                    "Speed-index generation was cancelled"
                )

        check_cancelled()
        initial_state = source_data_state()
        if log:
            log("Loading historical performances for speed-index generation")
        raw = _source_storage.load_records("historical")
        history = prepare_speed_performances(raw)
        targets = history.copy()
        generator = SpeedIndexGenerator(
            progress=progress,
            check_cancelled=check_cancelled,
        )
        context = FeatureContext(
            config.feature_name,
            config.feature_version,
            {"speed_index": {
                "half_life_days": config.half_life_days,
                "max_lookback_days": config.max_lookback_days,
                "seconds_scale_at_1600m": config.seconds_scale_at_1600m,
                "weight_points_per_kg": config.weight_points_per_kg,
            }},
        )
        generated = generator.transform(targets, history, context)
        generated["race_date"] = targets["race_date"].dt.date
        generated["performance_feature_name"] = config.feature_name
        generated["performance_feature_version"] = config.feature_version
        generated["history_cutoff"] = targets["race_date"]
        generated["feature_run_id"] = run_id
        generated["generated_at"] = generated_at
        final_state = source_data_state()
        check_cancelled()
        if final_state != initial_state:
            raise RuntimeError("Historical data changed during speed-index generation")
        if log:
            log("Saving per-performance speed indexes")
        replace_performance_features(generated, feature_database)
        completed = {
            **metadata,
            "status": "completed",
            "row_count": len(generated),
            "start_date": targets["race_date"].min().date(),
            "end_date": targets["race_date"].max().date(),
            "source_data_fingerprint": _fingerprint(raw),
            "source_state": final_state,
            "source_state_token": source_state_token(final_state),
            "message": "Speed-index generation completed",
        }
        save_feature_run(completed, feature_database)
        return completed
    except Exception as exc:
        status = (
            "cancelled"
            if isinstance(exc, SpeedIndexGenerationCancelled)
            else "failed"
        )
        save_feature_run({**metadata, "status": status, "message": str(exc)}, feature_database)
        raise
