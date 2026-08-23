from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .base import FeatureGenerator


_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class FeatureSetDefinition:
    """Versioned recipe describing exactly which generators produced a set."""

    name: str
    version: str
    generators: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _VERSION_PATTERN.fullmatch(self.name):
            raise ValueError(f"Invalid feature set name: {self.name!r}")
        if not _VERSION_PATTERN.fullmatch(self.version):
            raise ValueError(f"Invalid feature set version: {self.version!r}")
        if len(set(self.generators)) != len(self.generators):
            raise ValueError("Feature generator names must be unique")

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"


class FeatureRegistry:
    def __init__(self, generators: Iterable[FeatureGenerator] = ()) -> None:
        self._generators: dict[str, FeatureGenerator] = {}
        for generator in generators:
            self.register(generator)

    def register(self, generator: FeatureGenerator) -> None:
        if not generator.name or not _VERSION_PATTERN.fullmatch(generator.version):
            raise ValueError("Generator name and version must be non-empty and version-safe")
        if generator.name in self._generators:
            raise ValueError(f"Feature generator already registered: {generator.name}")
        duplicated = set(generator.output_columns).intersection(self.output_columns)
        if duplicated:
            raise ValueError(f"Feature columns already registered: {sorted(duplicated)}")
        self._generators[generator.name] = generator

    @property
    def output_columns(self) -> set[str]:
        return {
            column
            for generator in self._generators.values()
            for column in generator.output_columns
        }

    def resolve(self, definition: FeatureSetDefinition) -> tuple[FeatureGenerator, ...]:
        missing = set(definition.generators).difference(self._generators)
        if missing:
            raise ValueError(f"Unknown feature generators: {sorted(missing)}")
        return tuple(self._generators[name] for name in definition.generators)
