from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FeatureContext:
    """Immutable settings shared by all generators in one feature run."""

    feature_set_name: str
    feature_set_version: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        return f"{self.feature_set_name}:{self.feature_set_version}"
