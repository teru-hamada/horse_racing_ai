"""Point-in-time feature generation and persistence infrastructure."""

from .base import FeatureGenerator
from .context import FeatureContext
from .common import (
    calculate_elapsed_days,
    decay_weight,
    decay_weights,
    effective_count,
    weighted_mean,
    weighted_sum,
)
from .pipeline import FeaturePipeline
from .registry import FeatureRegistry, FeatureSetDefinition
from .generators import (
    RaceEntryGenerator,
    RecentFormGenerator,
    RecentSpeedGenerator,
    SpeedIndexGenerator,
    prepare_historical_performances,
    prepare_race_entries,
    prepare_speed_performances,
)
from .freshness import (
    feature_freshness,
    performance_feature_freshness,
    recent_speed_freshness,
    source_data_state,
    source_state_token,
)
from .service import (
    FeatureGenerationCancelled,
    RecentFormRunConfig,
    generate_recent_form_features,
)
from .speed_service import SpeedIndexRunConfig, generate_speed_index_features
from .recent_speed_service import (
    RecentSpeedGenerationCancelled,
    RecentSpeedRunConfig,
    generate_recent_speed_features,
)
from .race_entry_service import (
    RaceEntryGenerationCancelled,
    RaceEntryRunConfig,
    generate_race_entry_features,
)

__all__ = [
    "FeatureContext",
    "FeatureGenerator",
    "FeatureGenerationCancelled",
    "FeaturePipeline",
    "FeatureRegistry",
    "FeatureSetDefinition",
    "feature_freshness",
    "performance_feature_freshness",
    "recent_speed_freshness",
    "RaceEntryGenerator",
    "RaceEntryGenerationCancelled",
    "RaceEntryRunConfig",
    "RecentFormGenerator",
    "RecentFormRunConfig",
    "RecentSpeedGenerator",
    "RecentSpeedGenerationCancelled",
    "RecentSpeedRunConfig",
    "SpeedIndexGenerator",
    "SpeedIndexRunConfig",
    "calculate_elapsed_days",
    "decay_weight",
    "decay_weights",
    "effective_count",
    "weighted_mean",
    "weighted_sum",
    "prepare_historical_performances",
    "prepare_race_entries",
    "prepare_speed_performances",
    "generate_recent_form_features",
    "generate_race_entry_features",
    "generate_recent_speed_features",
    "generate_speed_index_features",
    "source_data_state",
    "source_state_token",
]
