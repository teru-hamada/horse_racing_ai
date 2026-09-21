from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from .feature_engineering.freshness import feature_freshness, recent_speed_freshness
from .feature_engineering.storage import (
    FEATURE_KEY_COLUMNS,
    feature_store_summary,
    load_features,
)


JOIN_KEYS = ("race_id", "horse_id")
DEFAULT_METADATA_COLUMNS = ("race_id", "horse_id", "race_date")


@dataclass(frozen=True)
class TrainingFeatureSet:
    name: str
    version: str

    @classmethod
    def parse(cls, identifier: str) -> "TrainingFeatureSet":
        name, separator, version = identifier.partition(":")
        if not separator or not name or not version:
            raise ValueError(
                f"Feature-set identifier must have the form name:version: {identifier!r}"
            )
        return cls(name=name, version=version)

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"


@dataclass(frozen=True)
class TrainingDatasetConfig:
    feature_sets: tuple[str, ...] = (
        "baseline:1.1.0",
        "recent_speed:1.0.0",
        "race_entry:1.1.0",
    )
    target_column: str = "target_top3"
    base_feature_columns: tuple[str, ...] = ()
    metadata_columns: tuple[str, ...] = DEFAULT_METADATA_COLUMNS
    start_date: date | None = None
    end_date: date | None = None
    validation_start_date: date | None = None
    test_start_date: date | None = None
    add_missing_indicators: bool = True
    require_fresh_features: bool = True
    require_complete_coverage: bool = False


@dataclass(frozen=True)
class FeatureSetBuildReport:
    identifier: str
    feature_run_id: str | None
    freshness_status: str
    feature_columns: tuple[str, ...]
    source_rows: int
    matched_rows: int
    missing_rows: int


@dataclass
class TrainingDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str
    metadata_columns: tuple[str, ...]
    reports: tuple[FeatureSetBuildReport, ...]
    manifest: dict[str, object]

    @property
    def X(self) -> pd.DataFrame:
        return self.frame.loc[:, list(self.feature_columns)].copy()

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target_column].copy()

    @property
    def metadata(self) -> pd.DataFrame:
        columns = [column for column in self.metadata_columns if column in self.frame]
        return self.frame.loc[:, columns].copy()


FreshnessChecker = Callable[..., Mapping[str, object]]


class TrainingDatasetBuilder:
    """Build a point-in-time training table from versioned feature sets.

    Design contract for every current and future feature set:

    * The feature generator, not this builder, must expand jockey, trainer,
      pedigree, race-level, or any other source grain to exactly one row per
      ``race_id + horse_id`` before saving it to ``features.duckdb``.
    * A feature set must be versioned, freshness-checkable, and contain no
      duplicate ``race_id + horse_id`` keys.
    * Feature column names must be unique across sets (a set-specific prefix is
      recommended).

    With this contract, adding jockey or pedigree features only adds another
    ``name:version`` to ``TrainingDatasetConfig.feature_sets``; the join logic
    in this builder does not change.

    前提: 騎手・調教師・血統・レース単位など、元の集計粒度にかかわらず、
    特徴量生成側が保存前に必ず race_id + horse_id の1行へ展開すること。
    この前提を守る限り、新しい特徴量を追加してもビルダー本体は変更しない。
    """

    def __init__(
        self,
        *,
        feature_database: Path | None = None,
        source_database: Path | None = None,
        freshness_checkers: Mapping[str, FreshnessChecker] | None = None,
    ) -> None:
        self.feature_database = feature_database
        self.source_database = source_database
        self.freshness_checkers = dict(freshness_checkers or {})
        self.freshness_checkers.setdefault("recent_speed:1.0.0", recent_speed_freshness)

    def build(
        self,
        base_frame: pd.DataFrame,
        config: TrainingDatasetConfig = TrainingDatasetConfig(),
    ) -> TrainingDataset:
        """Join only explicitly selected pre-race columns to the target rows."""

        feature_sets = tuple(TrainingFeatureSet.parse(item) for item in config.feature_sets)
        if len({item.identifier for item in feature_sets}) != len(feature_sets):
            raise ValueError("Feature sets must not be selected more than once")
        self._validate_config(config)
        required = set(JOIN_KEYS) | {"race_date", config.target_column}
        required.update(config.metadata_columns)
        required.update(config.base_feature_columns)
        missing = required.difference(base_frame.columns)
        if missing:
            raise ValueError(f"Training base is missing columns: {sorted(missing)}")

        frame = base_frame.copy()
        frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
        if frame["race_date"].isna().any():
            raise ValueError("Training base contains invalid race_date values")
        if frame.duplicated(list(JOIN_KEYS)).any():
            raise ValueError("Training base contains duplicate race_id + horse_id keys")
        if config.start_date is not None:
            frame = frame[frame["race_date"].dt.date >= config.start_date]
        if config.end_date is not None:
            frame = frame[frame["race_date"].dt.date <= config.end_date]
        if frame.empty:
            raise ValueError("No training rows exist in the requested period")

        keep = list(dict.fromkeys(
            [*config.metadata_columns, config.target_column, *config.base_feature_columns]
        ))
        frame = frame.loc[:, keep]
        model_columns = list(config.base_feature_columns)
        reports: list[FeatureSetBuildReport] = []
        used_columns = set(frame.columns)

        for feature_set in feature_sets:
            freshness = self._freshness(feature_set)
            status = str(freshness.get("status", "unknown"))
            if config.require_fresh_features and status != "fresh":
                raise ValueError(
                    f"Feature set {feature_set.identifier} is not fresh: {status}"
                )
            stored = load_features(
                feature_set.name, feature_set.version, self.feature_database
            )
            if stored.empty:
                raise ValueError(f"Feature set is not generated: {feature_set.identifier}")
            if stored.duplicated(list(JOIN_KEYS)).any():
                raise ValueError(
                    f"Feature set contains duplicate race_id + horse_id keys: "
                    f"{feature_set.identifier}"
                )

            candidate_columns = [
                column
                for column in stored.columns
                if column not in FEATURE_KEY_COLUMNS and stored[column].notna().any()
            ]
            if not candidate_columns:
                raise ValueError(
                    f"Feature set has no populated feature columns: {feature_set.identifier}"
                )
            collisions = used_columns.intersection(candidate_columns)
            if collisions:
                raise ValueError(
                    f"Feature columns collide for {feature_set.identifier}: "
                    f"{sorted(collisions)}"
                )

            join_frame = stored.loc[:, [*JOIN_KEYS, *candidate_columns]].copy()
            marker = f"__matched_{len(reports)}"
            join_frame[marker] = True
            frame = frame.merge(
                join_frame, on=list(JOIN_KEYS), how="left", validate="one_to_one"
            )
            matched_rows = int(frame[marker].fillna(False).sum())
            missing_rows = len(frame) - matched_rows
            frame.drop(columns=marker, inplace=True)
            if config.require_complete_coverage and missing_rows:
                raise ValueError(
                    f"Feature set {feature_set.identifier} is missing {missing_rows:,} rows"
                )

            if config.add_missing_indicators:
                for column in candidate_columns:
                    if frame[column].isna().any():
                        indicator = f"{column}__missing"
                        if indicator in used_columns or indicator in candidate_columns:
                            raise ValueError(f"Missing-indicator column collides: {indicator}")
                        frame[indicator] = frame[column].isna().astype("int8")
                        model_columns.append(indicator)
                        used_columns.add(indicator)
            model_columns.extend(candidate_columns)
            used_columns.update(candidate_columns)
            summary = feature_store_summary(
                feature_set.name, feature_set.version, self.feature_database
            )
            reports.append(FeatureSetBuildReport(
                identifier=feature_set.identifier,
                feature_run_id=(
                    str(summary["latest_run_id"])
                    if summary.get("latest_run_id") is not None else None
                ),
                freshness_status=status,
                feature_columns=tuple(candidate_columns),
                source_rows=len(stored),
                matched_rows=matched_rows,
                missing_rows=missing_rows,
            ))

        frame = frame.sort_values(
            ["race_date", "race_id", "horse_id"], kind="stable"
        ).reset_index(drop=True)
        manifest = {
            "builder_version": "1.0.0",
            "feature_sets": [
                {
                    "identifier": report.identifier,
                    "feature_run_id": report.feature_run_id,
                    "freshness_status": report.freshness_status,
                    "feature_columns": list(report.feature_columns),
                    "matched_rows": report.matched_rows,
                    "missing_rows": report.missing_rows,
                }
                for report in reports
            ],
            "target_column": config.target_column,
            "feature_columns": model_columns,
            "start_date": frame["race_date"].min().date().isoformat(),
            "end_date": frame["race_date"].max().date().isoformat(),
            "row_count": len(frame),
            "race_count": int(frame["race_id"].nunique()),
        }
        return TrainingDataset(
            frame=frame,
            feature_columns=tuple(model_columns),
            target_column=config.target_column,
            metadata_columns=config.metadata_columns,
            reports=tuple(reports),
            manifest=manifest,
        )

    def split_by_date(
        self,
        dataset: TrainingDataset,
        config: TrainingDatasetConfig,
    ) -> dict[str, TrainingDataset]:
        """Split on date boundaries so one race day never crosses partitions."""

        if config.validation_start_date is None or config.test_start_date is None:
            raise ValueError("validation_start_date and test_start_date are required")
        if config.validation_start_date >= config.test_start_date:
            raise ValueError("validation_start_date must be before test_start_date")
        dates = dataset.frame["race_date"].dt.date
        masks = {
            "train": dates < config.validation_start_date,
            "validation": (
                (dates >= config.validation_start_date) & (dates < config.test_start_date)
            ),
            "test": dates >= config.test_start_date,
        }
        result: dict[str, TrainingDataset] = {}
        for name, mask in masks.items():
            part = dataset.frame.loc[mask].reset_index(drop=True)
            if part.empty:
                raise ValueError(f"Training split is empty: {name}")
            result[name] = TrainingDataset(
                frame=part,
                feature_columns=dataset.feature_columns,
                target_column=dataset.target_column,
                metadata_columns=dataset.metadata_columns,
                reports=dataset.reports,
                manifest={**dataset.manifest, "partition": name, "row_count": len(part)},
            )
        return result

    def _freshness(self, feature_set: TrainingFeatureSet) -> Mapping[str, object]:
        checker = self.freshness_checkers.get(feature_set.identifier, feature_freshness)
        if checker is feature_freshness:
            return checker(
                feature_set.name,
                feature_set.version,
                source_database=self.source_database,
                feature_database=self.feature_database,
            )
        return checker(feature_database=self.feature_database)

    @staticmethod
    def _validate_config(config: TrainingDatasetConfig) -> None:
        if config.start_date and config.end_date and config.start_date > config.end_date:
            raise ValueError("start_date must not be after end_date")
        if config.target_column in config.base_feature_columns:
            raise ValueError("The target column must not be included as a base feature")
        if set(JOIN_KEYS).difference(config.metadata_columns):
            raise ValueError("metadata_columns must include race_id and horse_id")
