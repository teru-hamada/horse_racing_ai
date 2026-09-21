from __future__ import annotations

import pandas as pd

from .base import FEATURE_ID_COLUMNS
from .context import FeatureContext
from .registry import FeatureRegistry, FeatureSetDefinition


class FeaturePipeline:
    def __init__(self, registry: FeatureRegistry) -> None:
        self.registry = registry

    def transform(
        self,
        targets: pd.DataFrame,
        history: pd.DataFrame,
        definition: FeatureSetDefinition,
    ) -> pd.DataFrame:
        missing = set(FEATURE_ID_COLUMNS).difference(targets.columns)
        if missing:
            raise ValueError(f"Targets are missing feature IDs: {sorted(missing)}")

        context = FeatureContext(
            feature_set_name=definition.name,
            feature_set_version=definition.version,
            parameters=definition.parameters,
        )
        result = targets.loc[:, FEATURE_ID_COLUMNS].reset_index(drop=True).copy()
        for generator in self.registry.resolve(definition):
            generated = generator.transform(targets, history, context).reset_index(drop=True)
            generator.validate_output(targets.reset_index(drop=True), generated)
            result = pd.concat(
                [result, generated.loc[:, generator.output_columns]],
                axis=1,
            )
        return result
