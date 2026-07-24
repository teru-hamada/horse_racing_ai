from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import PATHS
from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, build_prediction_frame, build_training_frame
from .storage import json_dump, save_model_run, save_prediction_run


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else float("nan")


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(df["race_date"]).dropna().unique()))
    if len(dates) < 8:
        raise ValueError("時系列分割には少なくとも8開催日分の履歴データが必要です。")
    train_end = max(1, int(len(dates) * 0.70))
    val_end = max(train_end + 1, int(len(dates) * 0.85))
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])
    date_series = pd.to_datetime(df["race_date"])
    train = df[date_series.isin(train_dates)].copy()
    validation = df[date_series.isin(val_dates)].copy()
    test = df[date_series.isin(test_dates)].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("学習・検証・テストのいずれかが0件です。データ期間を増やしてください。")
    return train, validation, test


def _make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 0.001
    hidden_dim: int = 64
    dropout: float = 0.25
    patience: int = 12
    random_seed: int = 42


def train_model(
    historical_records: pd.DataFrame,
    config: TrainConfig,
    log: Callable[[str], None],
    progress: Callable[[float], None] | None = None,
) -> dict[str, object]:
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    model_run_id = f"model_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    model_dir = PATHS.models / model_run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    original_rows = len(historical_records)
    original_races = (
        historical_records["race_id"].nunique()
        if "race_id" in historical_records.columns
        else 0
    )

    log(
        f"学習元データ: {original_rows:,}行 / "
        f"{original_races:,}レース"
    )

    training = build_training_frame(historical_records, log=log)

    processed_rows = len(training)
    removed_rows = max(original_rows - processed_rows, 0)
    processed_races = (
        training["race_id"].nunique()
        if "race_id" in training.columns
        else 0
    )

    log(
        f"学習前処理後データ: {processed_rows:,}行 / "
        f"{processed_races:,}レース"
    )
    log(f"前処理で除外された行数: {removed_rows:,}行")

    diagnostic_columns = list(
        dict.fromkeys(
            ["race_id", "race_date", "target_top3"]
            + MODEL_FEATURES
        )
    )

    missing_columns = [
        column
        for column in diagnostic_columns
        if column not in training.columns
    ]
    if missing_columns:
        log(
            "前処理後データに存在しない列: "
            + ", ".join(missing_columns)
        )

    available_columns = [
        column
        for column in diagnostic_columns
        if column in training.columns
    ]

    if available_columns:
        missing_counts = (
            training[available_columns]
            .isna()
            .sum()
            .sort_values(ascending=False)
        )
        for column, count in missing_counts.items():
            if int(count) > 0:
                ratio = int(count) / max(processed_rows, 1)
                log(
                    f"欠損: {column}={int(count):,}行 "
                    f"({ratio:.1%})"
                )

    if processed_rows < 500:
        raise ValueError(
            "学習前処理後の有効データが"
            f"{processed_rows:,}行しかありません。"
            f"元データは{original_rows:,}行、"
            f"前処理で{removed_rows:,}行が除外されました。"
            "実行ログの欠損列・不足列を確認してください。"
        )

    train_df, val_df, test_df = _chronological_split(training)
    log(f"時系列分割: 学習={len(train_df)}、検証={len(val_df)}、テスト={len(test_df)}")

    preprocessor = _make_preprocessor()
    x_train = preprocessor.fit_transform(train_df[MODEL_FEATURES]).astype("float32")
    x_val = preprocessor.transform(val_df[MODEL_FEATURES]).astype("float32")
    x_test = preprocessor.transform(test_df[MODEL_FEATURES]).astype("float32")
    y_train = train_df["target_top3"].to_numpy(dtype="float32")
    y_val = val_df["target_top3"].to_numpy(dtype="float32")
    y_test = test_df["target_top3"].to_numpy(dtype="float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"学習デバイス: {device}")
    model = MLP(x_train.shape[1], hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    x_train_t = torch.from_numpy(x_train).to(device)

    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_epoch = 0
    waiting = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(x_batch)
            seen += len(x_batch)
        train_loss = total_loss / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            train_prob = torch.sigmoid(model(x_train_t)).cpu().numpy()
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        train_auc = _safe_auc(y_train, train_prob)
        val_auc = _safe_auc(y_val, val_prob)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "train_auc": train_auc,
                "validation_auc": val_auc,
            }
        )
        log(
            f"epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_auc={train_auc:.3f}, val_auc={val_auc:.3f}"
        )
        if progress:
            progress(epoch / config.epochs)

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            waiting = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= config.patience:
                log(f"Early Stopping: {epoch}エポックで停止")
                break

    if best_state is None:
        raise RuntimeError("学習済みモデルを保存できませんでした。")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(model(torch.from_numpy(x_test).to(device))).cpu().numpy()
    test_pred = (test_prob >= 0.5).astype(int)
    metrics = {
        "model_run_id": model_run_id,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "validation_auc": float(history[best_epoch - 1]["validation_auc"]),
        "test_auc": _safe_auc(y_test, test_prob),
        "test_log_loss": float(log_loss(y_test, np.clip(test_prob, 1e-6, 1 - 1e-6))),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "input_dim": int(x_train.shape[1]),
        "device": str(device),
        "features": MODEL_FEATURES,
        "config": config.__dict__,
        "date_ranges": {
            "train": [str(train_df["race_date"].min()), str(train_df["race_date"].max())],
            "validation": [str(val_df["race_date"].min()), str(val_df["race_date"].max())],
            "test": [str(test_df["race_date"].min()), str(test_df["race_date"].max())],
        },
    }

    torch.save(
        {
            "state_dict": best_state,
            "input_dim": x_train.shape[1],
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
        },
        model_dir / "model.pt",
    )
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "race_id": test_df["race_id"].to_numpy(),
            "horse_id": test_df["horse_id"].to_numpy(),
            "horse_name": test_df["horse_name"].to_numpy(),
            "actual_top3": y_test.astype(int),
            "predicted_probability": test_prob,
        }
    ).to_parquet(model_dir / "test_predictions.parquet", index=False)
    json_dump(model_dir / "metrics.json", metrics)

    save_model_run(
        {
            "model_run_id": model_run_id,
            "created_at": datetime.now(),
            "status": "completed",
            "target": "top3",
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "best_epoch": best_epoch,
            "validation_auc": metrics["validation_auc"],
            "test_auc": metrics["test_auc"],
            "test_log_loss": metrics["test_log_loss"],
            "model_path": str(model_dir),
            "metrics_json": json.dumps(metrics, ensure_ascii=False, default=str),
        }
    )
    log(f"モデル保存完了: {model_dir}")
    return metrics


def load_model_bundle(model_dir: Path) -> tuple[MLP, ColumnTransformer, dict[str, object]]:
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model = MLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    return model, preprocessor, metrics



def predict_historical_race(
    historical_records: pd.DataFrame,
    race_id: str,
    model_dir: Path,
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    """
    過去の指定レースを、対象レースより前の履歴だけで特徴量生成して予想し、
    実際の着順と比較する。

    注意:
        保存済みモデル自体が対象レースを学習に使用している場合、
        完全な未見データ検証にはならない。
    """
    race_id = str(race_id)
    records = historical_records.copy()
    records["race_id"] = records["race_id"].astype(str)
    records["race_date"] = pd.to_datetime(
        records["race_date"],
        errors="coerce",
    )

    actual = records[records["race_id"] == race_id].copy()
    if actual.empty:
        raise ValueError("指定した過去レースのデータがありません。")

    target_date = actual["race_date"].dropna()
    if target_date.empty:
        raise ValueError("指定レースの開催日を取得できません。")
    target_date = target_date.iloc[0]

    # 対象レース当日以降の履歴を除外し、未来情報の混入を防ぐ。
    history_before = records[
        records["race_date"] < target_date
    ].copy()
    if history_before.empty:
        raise ValueError(
            "対象レースより前の履歴データがありません。"
        )

    # 出走時点で未知の結果列を予想入力から除去する。
    upcoming_like = actual.copy()
    for column in [
        "finish_position",
        "time_seconds",
        "dataset_type",
        "collection_run_id",
        "collected_at",
    ]:
        if column in upcoming_like.columns:
            if column in {"finish_position", "time_seconds"}:
                upcoming_like[column] = np.nan
            elif column == "dataset_type":
                upcoming_like[column] = "upcoming"

    model, preprocessor, metrics = load_model_bundle(model_dir)
    featured = build_prediction_frame(
        history_before,
        upcoming_like,
    )
    target = featured[
        featured["race_id"].astype(str) == race_id
    ].copy()
    if target.empty:
        raise ValueError(
            "指定レースの予想用特徴量を生成できませんでした。"
        )

    matrix = preprocessor.transform(
        target[MODEL_FEATURES]
    ).astype("float32")
    with torch.no_grad():
        probability = torch.sigmoid(
            model(torch.from_numpy(matrix))
        ).numpy()

    result_columns = [
        "race_id",
        "race_date",
        "course_name",
        "race_number",
        "race_name",
        "horse_number",
        "horse_id",
        "horse_name",
        "jockey_name",
        "odds",
        "popularity",
    ]
    result = target[result_columns].copy()
    result["top3_probability"] = probability
    result["expected_value_index"] = (
        result["top3_probability"]
        * pd.to_numeric(result["odds"], errors="coerce")
    )

    actual_columns = [
        "horse_id",
        "horse_number",
        "finish_position",
    ]
    actual_result = actual[actual_columns].copy()
    actual_result["finish_position"] = pd.to_numeric(
        actual_result["finish_position"],
        errors="coerce",
    )

    merge_keys = ["horse_id"]
    if actual_result["horse_id"].isna().all():
        merge_keys = ["horse_number"]

    result = result.merge(
        actual_result,
        on=merge_keys,
        how="left",
        suffixes=("", "_actual"),
    )

    if "horse_number_actual" in result.columns:
        result["horse_number"] = result["horse_number"].fillna(
            result["horse_number_actual"]
        )
        result = result.drop(
            columns=["horse_number_actual"],
            errors="ignore",
        )

    result = result.sort_values(
        "top3_probability",
        ascending=False,
    ).reset_index(drop=True)
    result.insert(
        0,
        "prediction_rank",
        np.arange(1, len(result) + 1),
    )
    result["predicted_top3"] = (
        result["prediction_rank"] <= 3
    )
    result["actual_top3"] = (
        result["finish_position"] <= 3
    )
    result["top3_hit"] = (
        result["predicted_top3"]
        & result["actual_top3"]
    )

    predicted_top3_horses = set(
        result.loc[
            result["predicted_top3"],
            "horse_id",
        ].dropna().astype(str)
    )
    actual_top3_horses = set(
        result.loc[
            result["actual_top3"],
            "horse_id",
        ].dropna().astype(str)
    )
    top3_hit_count = len(
        predicted_top3_horses & actual_top3_horses
    )

    actual_winner = result[
        result["finish_position"] == 1
    ]
    winner_prediction_rank = (
        int(actual_winner["prediction_rank"].iloc[0])
        if not actual_winner.empty
        else None
    )

    summary = {
        "race_id": race_id,
        "race_date": str(target_date.date()),
        "history_rows_used": len(history_before),
        "top3_hit_count": top3_hit_count,
        "top3_hit_rate": top3_hit_count / 3.0,
        "prediction_1_actual_finish": (
            int(result.iloc[0]["finish_position"])
            if not result.empty
            and pd.notna(result.iloc[0]["finish_position"])
            else None
        ),
        "winner_prediction_rank": winner_prediction_rank,
        "model_run_id": metrics.get("model_run_id"),
    }

    prediction_run_id = (
        f"backtest_{datetime.now():%Y%m%d_%H%M%S}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    output_dir = PATHS.processed / prediction_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"historical_prediction_{race_id}.parquet"
    )
    result.to_parquet(output_path, index=False)
    result.to_csv(
        output_dir / f"historical_prediction_{race_id}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    json_dump(
        output_dir / f"historical_prediction_{race_id}_summary.json",
        summary,
    )

    save_prediction_run(
        {
            "prediction_run_id": prediction_run_id,
            "created_at": datetime.now(),
            "model_run_id": metrics["model_run_id"],
            "race_id": race_id,
            "row_count": len(result),
            "output_path": str(output_path),
        }
    )
    return result, output_path, summary

def predict_race(
    historical_records: pd.DataFrame,
    upcoming_records: pd.DataFrame,
    race_id: str,
    model_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    model, preprocessor, metrics = load_model_bundle(model_dir)
    featured = build_prediction_frame(historical_records, upcoming_records)
    target = featured[featured["race_id"].astype(str) == str(race_id)].copy()
    if target.empty:
        raise ValueError("指定したレースの出走データがありません。")
    matrix = preprocessor.transform(target[MODEL_FEATURES]).astype("float32")
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.from_numpy(matrix))).numpy()
    result = target[
        ["race_id", "race_date", "course_name", "race_number", "race_name", "horse_number", "horse_id", "horse_name", "jockey_name", "odds", "popularity"]
    ].copy()
    result["top3_probability"] = probability
    result["expected_value_index"] = result["top3_probability"] * pd.to_numeric(result["odds"], errors="coerce")
    result = result.sort_values("top3_probability", ascending=False).reset_index(drop=True)
    result.insert(0, "prediction_rank", np.arange(1, len(result) + 1))

    prediction_run_id = f"pred_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    output_dir = PATHS.processed / prediction_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"prediction_{race_id}.parquet"
    result.to_parquet(output_path, index=False)
    result.to_csv(output_dir / f"prediction_{race_id}.csv", index=False, encoding="utf-8-sig")
    save_prediction_run(
        {
            "prediction_run_id": prediction_run_id,
            "created_at": datetime.now(),
            "model_run_id": metrics["model_run_id"],
            "race_id": race_id,
            "row_count": len(result),
            "output_path": str(output_path),
        }
    )
    return result, output_path
