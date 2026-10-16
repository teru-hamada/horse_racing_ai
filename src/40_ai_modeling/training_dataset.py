# 特徴量DBの選択済みセットをrace_id＋horse_idで結合し、学習表と構成記録を作る。
# 列名衝突・重複キー・鮮度・不足行を検査し、目的変数や管理用列が入力特徴量へ混ざるのを防ぐ。
# 学習・検証・テストの境界を明示日付で指定でき、同じ開催日の行を別区分へ分散させない。
#
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from .feature_engineering.freshness import feature_freshness, recent_speed_freshness
from .feature_engineering.storage import (
    FEATURE_KEY_COLUMNS,
    feature_store_summary,
    load_features,
)


# 学習対象と特徴量DBを結ぶキー。一レース一頭が一行となる粒度を保証する。
JOIN_KEYS = ("race_id", "horse_id")
# モデル入力と分離して保持する既定のID・日付列。
DEFAULT_METADATA_COLUMNS = ("race_id", "horse_id", "race_date")


# 学習で選択する特徴量セットを名前と版の組で表し、name:version形式と相互変換する。
@dataclass(frozen=True)
class TrainingFeatureSet:
    # 登録・保存に使う内部名。画面ラベルとは別に扱う。
    name: str
    # この生成器または定義の版。計算内容を変更するときの識別に使う。
    version: str

    # 文字列のname:versionをセット指定へ変換し、構成を曖昧にしない。
    @classmethod
    def parse(cls, identifier: str) -> "TrainingFeatureSet":
        name, separator, version = identifier.partition(":")
        if not separator or not name or not version:
            raise ValueError(
                f"Feature-set identifier must have the form name:version: {identifier!r}"
            )
        return cls(name=name, version=version)

    # 名前と版をコロンで連結した識別子。保存・設定・照合で同じ形式を使う。
    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"


# 結合するセット、入力列、期間、欠損・鮮度の許容条件を指定する学習表構築用の設定。
@dataclass(frozen=True)
class TrainingDatasetConfig:
    # 結合するname:versionの一覧。順序も入力特徴量の構成へ反映される。
    feature_sets: tuple[str, ...] = (
        "baseline:1.1.0",
        "recent_speed:1.0.0",
        "race_entry:1.1.0",
    )
    # 正解ラベルの列名。入力特徴量Xへ含めず、目的変数yとして取り出す。
    target_column: str = "target_top3"
    # 特徴量DBとの結合前から入力に使うことを明示的に許可する列。
    base_feature_columns: tuple[str, ...] = ()
    # ID・日付など結果対応付け用の列。モデル入力とは別に保持する。
    metadata_columns: tuple[str, ...] = DEFAULT_METADATA_COLUMNS
    # 処理対象期間の開始日。Noneを許容する設定では開始側を制限しない。
    start_date: date | None = None
    # 処理対象期間の終了日。Noneを許容する設定では終了側を制限しない。
    end_date: date | None = None
    # この日以上を検証区分とする境界。テスト開始日より前であることが必要。
    validation_start_date: date | None = None
    # この日以上をテスト区分にする境界。検証区分と重複させない。
    test_start_date: date | None = None
    # 欠損のある特徴量へ__missing列を追加し、欠測だった事実を学習可能にする。
    add_missing_indicators: bool = True
    # Trueなら生成元状態がfreshと確認できないセットの学習利用を拒否する。
    require_fresh_features: bool = True
    # Trueなら学習対象に対応する特徴量行が一つでも欠けたときに構築を止める。
    require_complete_coverage: bool = False


# 一つのセットの結合実績。使った版・生成run・列と、一致／不足行数を記録する。
@dataclass(frozen=True)
class FeatureSetBuildReport:
    identifier: str
    # 保存された特徴量を生成した実行ID。どの計算結果を使用したかを追跡する。
    feature_run_id: str | None
    # 結合前の鮮度判定。fresh・stale・missing・unknownなどを記録する。
    freshness_status: str
    # モデルへ入力する列名と順序。識別列や目的変数とは分けて管理する。
    feature_columns: tuple[str, ...]
    # 特徴量DBから読み出した、そのセット・版の行数。
    source_rows: int
    # 学習対象キーに対応する特徴量行が見つかった件数。
    matched_rows: int
    # 対象キーに対応する特徴量行がなかった件数。列単位のNaN数とは異なる。
    missing_rows: int


# 特徴量・正解・管理列を含む表と、その列定義・構築レポート・manifestをまとめた戻り値。
@dataclass
class TrainingDataset:
    # 正解・管理情報・入力特徴量を同じ行順で持つ学習用の表。
    frame: pd.DataFrame
    # モデルへ入力する列名と順序。識別列や目的変数とは分けて管理する。
    feature_columns: tuple[str, ...]
    # 正解ラベルの列名。入力特徴量Xへ含めず、目的変数yとして取り出す。
    target_column: str
    # ID・日付など結果対応付け用の列。モデル入力とは別に保持する。
    metadata_columns: tuple[str, ...]
    # セットごとの結合件数と鮮度・使用列の一覧。
    reports: tuple[FeatureSetBuildReport, ...]
    # 利用した版、入力列、対象期間、件数などを成果物へ残すための構成情報。
    manifest: dict[str, object]

    # モデル入力列だけをコピーして返し、正解ラベルや管理用列を入力から分離する。
    @property
    def X(self) -> pd.DataFrame:
        return self.frame.loc[:, list(self.feature_columns)].copy()

    # 設定した目的変数列を、入力特徴量と同じ行順で取り出す。
    @property
    def y(self) -> pd.Series:
        return self.frame[self.target_column].copy()

    # 予想結果との対応付けに使うID・日付などを、入力特徴量とは別に返す。
    @property
    def metadata(self) -> pd.DataFrame:
        columns = [column for column in self.metadata_columns if column in self.frame]
        return self.frame.loc[:, columns].copy()


FreshnessChecker = Callable[..., Mapping[str, object]]


# 指定された特徴量DBから学習表を構築する。鮮度確認の実装はセット単位で差し替えられる。
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

    # DB保存先とセット別の鮮度判定を保持し、近走速度には依存指数も調べる判定を既定登録する。
    def __init__(
        self,
        *,
        feature_database: Path | None = None,
        source_database: Path | None = None,
        freshness_checkers: Mapping[str, FreshnessChecker] | None = None,
    ) -> None:
        # feature_database: 特徴量DBの指定先。省略可能な場合は既定のfeatures.duckdbを使う。
        # source_database: 鮮度確認などで参照する元レースDBの指定先。
        # 特徴量の読み書き先。Noneを許容するAPIではプロジェクトの既定DBを使用する。
        self.feature_database = feature_database
        # 鮮度確認などで参照する元レースDB。
        self.source_database = source_database
        # セット識別子ごとの鮮度判定関数。依存関係のあるセットに専用判定を指定できる。
        self.freshness_checkers = dict(freshness_checkers or {})
        self.freshness_checkers.setdefault("recent_speed:1.0.0", recent_speed_freshness)

    # 選択セットの鮮度・キー・列名を検査して左結合する。不足行は設定に応じて拒否または欠損フラグを付ける。
    def build(
        self,
        base_frame: pd.DataFrame,
        config: TrainingDatasetConfig = TrainingDatasetConfig(),
    ) -> TrainingDataset:
        """Join only explicitly selected pre-race columns to the target rows."""

        # 設定に指定されたセット名と版を解釈した順序付き一覧。
        feature_sets = tuple(TrainingFeatureSet.parse(item) for item in config.feature_sets)
        if len({item.identifier for item in feature_sets}) != len(feature_sets):
            raise ValueError("Feature sets must not be selected more than once")
        self._validate_config(config)
        required = set(JOIN_KEYS) | {"race_date", config.target_column}
        required.update(config.metadata_columns)
        required.update(config.base_feature_columns)
        missing = required.difference(base_frame.columns)
        # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
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

        # ベース表から保持する列の許可リスト。結果由来の余分な列を自動採用しない。
        keep = list(dict.fromkeys(
            [*config.metadata_columns, config.target_column, *config.base_feature_columns]
        ))
        frame = frame.loc[:, keep]
        # 入力Xへ入れる列の順序を蓄積する一覧。欠損フラグもここへ追加する。
        model_columns = list(config.base_feature_columns)
        reports: list[FeatureSetBuildReport] = []
        # 結合済みの列名。別セットや欠損フラグとの名前衝突を検出する。
        used_columns = set(frame.columns)

        # 設定順にセットを結合する。各セットの管理列や重複列がモデル入力へ混ざらないよう検査する。
        for feature_set in feature_sets:
            freshness = self._freshness(feature_set)
            status = str(freshness.get("status", "unknown"))
            # 鮮度必須の設定ではunknownも許可しない。最新と確認できた特徴量だけを学習に使う。
            if config.require_fresh_features and status != "fresh":
                raise ValueError(
                    f"Feature set {feature_set.identifier} is not fresh: {status}"
                )
            # 指定したセット・版の保存済み特徴量。管理列と有効な入力候補列を切り分ける。
            stored = load_features(
                feature_set.name, feature_set.version, self.feature_database
            )
            # 選択したセット・版が生成されていなければ、全欠損のまま学習へ渡さない。
            if stored.empty:
                raise ValueError(f"Feature set is not generated: {feature_set.identifier}")
            if stored.duplicated(list(JOIN_KEYS)).any():
                raise ValueError(
                    f"Feature set contains duplicate race_id + horse_id keys: "
                    f"{feature_set.identifier}"
                )

            # 版や生成日時などの管理列と、その版で値が一件もない列はモデル入力から外す。
            candidate_columns = [
                column
                for column in stored.columns
                if column not in FEATURE_KEY_COLUMNS and stored[column].notna().any()
            ]
            # その版に値のある特徴量列がなければ、キーだけのセットとして利用することを拒否する。
            if not candidate_columns:
                raise ValueError(
                    f"Feature set has no populated feature columns: {feature_set.identifier}"
                )
            # 今回追加する列と既存列の重複。曖昧な結合を避けるため空である必要がある。
            collisions = used_columns.intersection(candidate_columns)
            # 別セットや管理列と名前が重なれば、mergeの接尾辞で曖昧にせず結合前に拒否する。
            if collisions:
                raise ValueError(
                    f"Feature columns collide for {feature_set.identifier}: "
                    f"{sorted(collisions)}"
                )

            # 対象キーと今回の特徴量だけを選んだ結合用の表。
            join_frame = stored.loc[:, [*JOIN_KEYS, *candidate_columns]].copy()
            # 結合成功の行数を数える一時列。特徴量自体がNaNでもキーが一致したかを区別できる。
            marker = f"__matched_{len(reports)}"
            join_frame[marker] = True
            # 学習対象の行を減らさない左結合。双方の一意性はone_to_oneでも検証する。
            frame = frame.merge(
                join_frame, on=list(JOIN_KEYS), how="left", validate="one_to_one"
            )
            matched_rows = int(frame[marker].fillna(False).sum())
            missing_rows = len(frame) - matched_rows
            frame.drop(columns=marker, inplace=True)
            # 全行被覆が必須なら、特徴量のキーが一行でも足りない場合に構築を止める。
            if config.require_complete_coverage and missing_rows:
                raise ValueError(
                    f"Feature set {feature_set.identifier} is missing {missing_rows:,} rows"
                )

            # 欠損値を後で補完しても、観測できなかった事実は別の入力列として学習に残す。
            if config.add_missing_indicators:
                for column in candidate_columns:
                    # 少なくとも一件欠損がある列だけに欠損フラグを作り、不要な定数列を増やさない。
                    if frame[column].isna().any():
                        # 欠損の有無を0／1で表す追加列名。補完後も欠測の情報を残す。
                        indicator = f"{column}__missing"
                        # 自動で作る欠損フラグも既存の特徴量と名前が衝突しないことを確認する。
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

    # 検証開始日より前を学習、検証開始日以上かつテスト開始日未満を検証、それ以降をテストとする。
    def split_by_date(
        self,
        dataset: TrainingDataset,
        config: TrainingDatasetConfig,
    ) -> dict[str, TrainingDataset]:
        """Split on date boundaries so one race day never crosses partitions."""

        if config.validation_start_date is None or config.test_start_date is None:
            raise ValueError("validation_start_date and test_start_date are required")
        # 検証期間が空または逆転しないよう、検証開始日はテスト開始日より前に限定する。
        if config.validation_start_date >= config.test_start_date:
            raise ValueError("validation_start_date must be before test_start_date")
        dates = dataset.frame["race_date"].dt.date
        # 学習・検証・テストの日付境界を表す互いに重複しない真偽マスク。
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
            # 分割後の一区分が空なら学習・評価ができないため、期間の見直しを要求する。
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

    # セット専用の鮮度チェッカーが登録されていれば使い、それ以外は元DB状態を比較する共通判定へ委ねる。
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

    # 対象期間の逆転、目的変数の特徴量への混入、結合キーの欠落を結合処理の前に拒否する。
    @staticmethod
    def _validate_config(config: TrainingDatasetConfig) -> None:
        if config.start_date and config.end_date and config.start_date > config.end_date:
            raise ValueError("start_date must not be after end_date")
        # 正解を入力として与える直接の情報漏洩を、設定検査の段階で拒否する。
        if config.target_column in config.base_feature_columns:
            raise ValueError("The target column must not be included as a base feature")
        if set(JOIN_KEYS).difference(config.metadata_columns):
            raise ValueError("metadata_columns must include race_id and horse_id")
