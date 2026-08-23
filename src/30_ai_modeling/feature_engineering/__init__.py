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
from .generators import RecentFormGenerator, prepare_historical_performances
from .service import (
    FeatureGenerationCancelled,
    RecentFormRunConfig,
    generate_recent_form_features,
)

__all__ = [
    "FeatureContext",
    "FeatureGenerator",
    "FeatureGenerationCancelled",
    "FeaturePipeline",
    "FeatureRegistry",
    "FeatureSetDefinition",
    "RecentFormGenerator",
    "RecentFormRunConfig",
    "calculate_elapsed_days",
    "decay_weight",
    "decay_weights",
    "effective_count",
    "weighted_mean",
    "weighted_sum",
    "prepare_historical_performances",
    "generate_recent_form_features",
]
