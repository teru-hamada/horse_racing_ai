from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .context import FeatureContext


FEATURE_ID_COLUMNS = ("race_id", "horse_id")


class FeatureGenerator(ABC):
    """Contract implemented by one independently versioned feature family."""

    name: str
    version: str
    output_columns: tuple[str, ...]

    @abstractmethod
    def transform(
        self,
        targets: pd.DataFrame,
        history: pd.DataFrame,
        context: FeatureContext,
    ) -> pd.DataFrame:
        """Return ID columns plus this generator's declared output columns."""

    def validate_output(
        self,
        targets: pd.DataFrame,
        output: pd.DataFrame,
    ) -> None:
        expected = set(FEATURE_ID_COLUMNS + self.output_columns)
        actual = set(output.columns)
        if actual != expected:
            raise ValueError(
                f"Feature generator {self.name!r} returned invalid columns: "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
        if output.duplicated(list(FEATURE_ID_COLUMNS)).any():
            raise ValueError(f"Feature generator {self.name!r} returned duplicate IDs")
        target_ids = targets.loc[:, FEATURE_ID_COLUMNS].reset_index(drop=True)
        output_ids = output.loc[:, FEATURE_ID_COLUMNS].reset_index(drop=True)
        if len(output_ids) != len(target_ids) or not output_ids.equals(target_ids):
            raise ValueError(
                f"Feature generator {self.name!r} must preserve target row order and IDs"
            )
