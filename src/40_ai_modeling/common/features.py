# 出走履歴から累積成績や前走からの日数を作る共通特徴量処理。
# 馬ごとに時系列順へ並べ、shiftで当該行の結果を集計対象から外す。
# 特徴量DBを使う現行Top3のセット構築とは別に、履歴ベースの共通APIを提供する。
#
from __future__ import annotations

from typing import Callable

import pandas as pd


# 履歴ベースの共通処理が宣言する数値入力列。現行の動的セット構成とは区別する。
NUMERIC_FEATURES = [
    "race_number", "distance", "horse_number", "frame_number", "age",
    "carried_weight", "odds", "popularity", "prior_starts",
    "prior_win_rate", "prior_top3_rate", "prior_avg_finish", "prior_avg_odds",
    "days_since_last_race",
]
# カテゴリとして扱う属性やIDの一覧。数値の大小として学習しない列。
CATEGORICAL_FEATURES = [
    "course_name", "surface", "weather", "track_condition", "sex",
    "jockey_id", "trainer_id", "sire_id", "dam_id", "damsire_id",
]
# 共通特徴量の数値列とカテゴリ列を連結した入力定義。
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# 馬ごとの並びを固定し、当該行より前の着順・オッズから累積成績を計算する。
def add_historical_features(
    records: pd.DataFrame,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if records.empty:
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log("特徴量生成対象データが0行です。")
        return records.copy()

    df = records.copy()

    # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
    if log:
        log(
            f"特徴量生成開始: {len(df):,}行 / "
            f"{len(df.columns):,}列"
        )
        log("入力列: " + ", ".join(map(str, df.columns)))

        # 列ごとの欠損数。前処理で学習行が不足した原因をログで確認する。
        missing_counts = df.isna().sum().sort_values(ascending=False)
        for column, count in missing_counts.items():
            if int(count) > 0:
                log(
                    f"入力欠損: {column}={int(count):,}行 "
                    f"({int(count) / max(len(df), 1):.1%})"
                )

    required_columns = [
        "race_date",
        "race_id",
        "horse_id",
        "horse_name",
        "horse_number",
        "finish_position",
        "odds",
    ]
    missing_required = [
        column for column in required_columns
        if column not in df.columns
    ]
    # 必須列不足は入力の構造エラー。欠損補完で隠さず、足りない列名を明示する。
    if missing_required:
        raise ValueError(
            "特徴量生成に必要な列がありません: "
            + ", ".join(missing_required)
        )

    original_race_date = df["race_date"].copy()
    original_finish = df["finish_position"].copy()
    original_odds = df["odds"].copy()

    df["race_date"] = pd.to_datetime(
        df["race_date"],
        errors="coerce",
    )
    df["finish_position"] = pd.to_numeric(
        df["finish_position"],
        errors="coerce",
    )
    df["odds"] = pd.to_numeric(
        df["odds"],
        errors="coerce",
    )

    # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
    if log:
        race_date_failed = int(
            original_race_date.notna().sum()
            - df["race_date"].notna().sum()
        )
        finish_failed = int(
            original_finish.notna().sum()
            - df["finish_position"].notna().sum()
        )
        odds_failed = int(
            original_odds.notna().sum()
            - df["odds"].notna().sum()
        )

        log(
            f"型変換失敗: race_date={race_date_failed:,}行、"
            f"finish_position={finish_failed:,}行、"
            f"odds={odds_failed:,}行"
        )

        finish_samples = (
            original_finish
            .dropna()
            .astype(str)
            .value_counts()
            .head(20)
        )
        if finish_samples.empty:
            log("finish_positionの元データは全行欠損です。")
        else:
            sample_text = ", ".join(
                f"{value}({count}件)"
                for value, count in finish_samples.items()
            )
            log(
                "finish_position元データの値例: "
                + sample_text
            )

        log(
            "変換後の有効件数: "
            f"race_date={df['race_date'].notna().sum():,}行、"
            f"finish_position={df['finish_position'].notna().sum():,}行、"
            f"odds={df['odds'].notna().sum():,}行"
        )

    df["horse_key"] = (
        df["horse_id"]
        .fillna(df["horse_name"])
        .astype(str)
    )
    df["is_win"] = (
        df["finish_position"] == 1
    ).astype(float)
    df["is_top3"] = (
        df["finish_position"] <= 3
    ).where(
        df["finish_position"].notna()
    ).astype(float)

    df = df.sort_values(
        ["horse_key", "race_date", "race_id", "horse_number"],
        kind="stable",
    ).reset_index(drop=True)

    group = df.groupby("horse_key", sort=False)
    df["prior_starts"] = group.cumcount().astype(float)
    df["prior_win_rate"] = group["is_win"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["prior_top3_rate"] = group["is_top3"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["prior_avg_finish"] = group["finish_position"].transform(
        lambda s: s.shift().expanding().mean()
    )
    df["prior_avg_odds"] = group["odds"].transform(
        lambda s: s.shift().expanding().mean()
    )
    previous_date = group["race_date"].shift()
    df["days_since_last_race"] = (
        df["race_date"] - previous_date
    ).dt.days

    fill_values = {
        "prior_win_rate": 0.0,
        "prior_top3_rate": 0.0,
        "prior_avg_finish": 9.0,
        "prior_avg_odds": 20.0,
        "days_since_last_race": 999.0,
    }
    df = df.fillna(fill_values)

    # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
    if log:
        log(
            f"特徴量生成完了: {len(df):,}行 / "
            f"着順有効={df['finish_position'].notna().sum():,}行"
        )

    return df.drop(
        columns=["horse_key", "is_win"],
        errors="ignore",
    )


# 特徴量生成後に着順が欠損した行を除き、除外件数をログへ残す。
def build_historical_training_base(
    records: pd.DataFrame,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    featured = add_historical_features(records, log=log)

    before_filter = len(featured)
    valid_finish = int(
        featured["finish_position"].notna().sum()
    )

    # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
    if log:
        log(
            f"着順フィルタ前: {before_filter:,}行 / "
            f"finish_position有効={valid_finish:,}行 / "
            f"欠損={before_filter - valid_finish:,}行"
        )

    featured = featured[
        featured["finish_position"].notna()
    ].copy()

    # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
    if log:
        log(
            f"着順フィルタ後: {len(featured):,}行"
        )
        if featured.empty:
            log(
                "全行除外の直接原因: "
                "finish_positionが全行欠損、"
                "または数値変換できない値です。"
            )

    return featured.sort_values(
        ["race_date", "race_id", "horse_number"]
    ).reset_index(drop=True)


# 過去と未開催の行を連結して履歴特徴量を作り、最後に未開催の行だけを取り出す。
def build_prediction_frame(historical: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    historical = historical.copy()
    upcoming = upcoming.copy()
    historical["dataset_type"] = "historical"
    upcoming["dataset_type"] = "upcoming"

    # Align all-null upcoming columns with the historical dtype before concatenation.
    # This keeps pandas behavior stable across versions (for example, finish_position).
    for column in set(historical.columns).intersection(upcoming.columns):
        if upcoming[column].isna().all():
            try:
                upcoming[column] = upcoming[column].astype(historical[column].dtype)
            except (TypeError, ValueError):
                pass
    combined = pd.concat([historical, upcoming], ignore_index=True, sort=False)
    featured = add_historical_features(combined)
    return featured[featured["dataset_type"] == "upcoming"].copy()
