# 3着以内を目的変数とするMLPの学習、保存、読み込み、各種予想を実装する。
# 特徴量DBから作った学習表を時系列分割し、学習区分だけで前処理器をfitする。
# 検証損失で早期終了と最良モデルを選び、テスト区分の指標・特徴量構成を成果物へ記録する。
# 過去レース予想では履歴の未来情報を除くが、モデル自体の学習期間とは別に注意が必要。
#
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# プロジェクト既定の保存先。DBを差し替えるテストではこの参照も隔離先へ変更する。
PATHS = import_module("src.00_common.config").PATHS
from ...common.datasets import chronological_split
from ...common.evaluation import safe_auc
from ...common.artifacts import save_manifest
from ...common.preprocessing import (
    infer_feature_types,
    make_registered_feature_preprocessor,
    prepare_registered_feature_frame,
)
from ...common.types import TrainConfig
from ...common.calibration import (
    ProbabilityCalibration, attach_probabilities, load_calibration, save_calibrations,
    save_comparison_outputs,
)
from ...estimators.mlp import MLP
from ...training_dataset import TrainingDatasetBuilder, TrainingDatasetConfig
from .config import (
    ESTIMATOR_NAME,
    FEATURE_VERSION,
    MODEL_VERSION,
    TARGET_COLUMN,
    TASK_NAME,
)
from .feature_set import align_saved_top3_features, build_top3_prediction_features

_storage = import_module("src.00_common.storage")
json_dump = _storage.json_dump
save_model_run = _storage.save_model_run
save_prediction_run = _storage.save_prediction_run


# 3着内ラベルを付けた表を構築し、前処理・MLP学習・評価・成果物保存・DB登録まで行う。
def train_model(
    historical_records: pd.DataFrame,
    config: TrainConfig,
    log: Callable[[str], None],
    progress: Callable[[float], None] | None = None,
) -> dict[str, object]:
    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # progress: 任意の進捗コールバック。値は通常0～1の完了率。
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    # 今回学習するモデルの一意なID。成果物フォルダーと登録DBに共通で記録する。
    model_run_id = f"model_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    model_dir = PATHS.models / "top3" / model_run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # 前処理前の行数。フィルターで減った件数を診断する基準。
    original_rows = len(historical_records)
    # 入力に含まれるレース数。出走馬の行数とは区別してログへ出す。
    original_races = (
        historical_records["race_id"].nunique()
        if "race_id" in historical_records.columns
        else 0
    )

    log(
        f"学習元データ: {original_rows:,}行 / "
        f"{original_races:,}レース"
    )

    # 正解ラベルを作る着順列がないデータでは、学習を開始しない。
    if "finish_position" not in historical_records:
        raise ValueError("学習元データにfinish_positionがありません。")
    # 有効な着順と目的変数を持つ、特徴量結合前の学習対象表。
    training_base = historical_records.copy()
    training_base["finish_position"] = pd.to_numeric(
        training_base["finish_position"], errors="coerce"
    )
    training_base = training_base[training_base["finish_position"].gt(0)].copy()
    training_base[TARGET_COLUMN] = training_base["finish_position"].le(3).astype("int8")
    if "horse_name" not in training_base:
        training_base["horse_name"] = pd.NA

    # 学習表と選択特徴量・結合レポート・構成記録をひとまとめにした結果。
    dataset = TrainingDatasetBuilder().build(
        training_base,
        TrainingDatasetConfig(
            target_column=TARGET_COLUMN,
            metadata_columns=("race_id", "horse_id", "race_date", "horse_name"),
        ),
    )
    training = dataset.frame
    # この学習runで採用する列の順序。推論時も同じ構成を使う。
    model_features = list(dataset.feature_columns)
    for report in dataset.reports:
        log(
            f"特徴量セット {report.identifier}: {len(report.feature_columns)}列 / "
            f"結合={report.matched_rows:,}行 / 欠損キー={report.missing_rows:,}行 / "
            f"鮮度={report.freshness_status}"
        )

    # 前処理と結合を終えた有効な学習行数。最低件数の検査に使う。
    processed_rows = len(training)
    # 前処理で失われた行数の診断値。入力件数との差から求める。
    removed_rows = max(original_rows - processed_rows, 0)
    # 前処理後の学習表に残ったレース数。
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

    # ID・開催日・正解と特徴量を合わせた欠損診断用の列一覧。
    diagnostic_columns = list(
        dict.fromkeys(
            ["race_id", "race_date", TARGET_COLUMN]
            + model_features
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
        # 列ごとの欠損数。前処理で学習行が不足した原因をログで確認する。
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

    # 前処理後の有効行数で最低件数を確認する。元データの件数だけでは判断しない。
    if processed_rows < 500:
        raise ValueError(
            "学習前処理後の有効データが"
            f"{processed_rows:,}行しかありません。"
            f"元データは{original_rows:,}行、"
            f"前処理で{removed_rows:,}行が除外されました。"
            "実行ログの欠損列・不足列を確認してください。"
        )

    # 開催日単位で分割し、同じ日の行が学習側と評価側の両方へ入らないようにする。
    train_df, val_df, test_df = chronological_split(training)
    log(f"時系列分割: 学習={len(train_df)}、検証={len(val_df)}、テスト={len(test_df)}")

    numeric_features, categorical_features = infer_feature_types(
        train_df, model_features
    )
    log(
        f"学習特徴量: 合計={len(model_features)}列 / "
        f"数値={len(numeric_features)}列 / カテゴリ={len(categorical_features)}列"
    )
    # 学習runに対応する補完・標準化・カテゴリ符号化の前処理器。
    preprocessor = make_registered_feature_preprocessor(
        numeric_features, categorical_features
    )
    # 補完値・標準化の統計量・カテゴリ辞書は学習区分だけから求める。
    # 検証とテストをfitへ含めると、将来データの分布が前処理へ漏れてしまう。
    x_train = preprocessor.fit_transform(
        prepare_registered_feature_frame(
            train_df, numeric_features, categorical_features
        )
    ).astype("float32")
    # 学習済み前処理器で変換した検証入力。前処理のfitには含めない。
    x_val = preprocessor.transform(
        prepare_registered_feature_frame(
            val_df, numeric_features, categorical_features
        )
    ).astype("float32")
    # 最終評価用の入力行列。前処理やエポック選択の学習対象にはしない。
    x_test = preprocessor.transform(
        prepare_registered_feature_frame(
            test_df, numeric_features, categorical_features
        )
    ).astype("float32")
    # 学習入力と同じ行順の二値正解ラベル。
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype="float32")
    # 検証区分の正解ラベル。損失監視と最良エポックの選択に使う。
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype="float32")
    # テスト区分の正解ラベル。選択済みモデルの評価にだけ使う。
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype="float32")

    # CUDAが利用可能ならGPU、それ以外はCPUを選ぶ実行先。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"学習デバイス: {device}")
    model = MLP(x_train.shape[1], hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    # 正例・負例の比率を損失の重みに反映する。logitを直接受ける損失なので、ここではsigmoidしない。
    positives = max(float(y_train.sum()), 1.0)
    # 学習負例数。正例数との比を損失関数のpos_weightへ渡す。
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    # クラス比率を補正する二値分類損失。確率ではなくlogitを受け取る。
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    # モデルのパラメーターを更新するAdamW。学習率と重み減衰を設定する。
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    # 学習行列と正解を対応付け、シャッフルしたミニバッチを取り出すローダー。
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    x_train_t = torch.from_numpy(x_train).to(device)

    history: list[dict[str, float | int]] = []
    # これまでの最良の検証損失。小さいほど良い。
    best_loss = math.inf
    # 最良検証損失を達成したエポック番号。保存指標と履歴の対応に使う。
    best_epoch = 0
    # 検証損失が十分に改善しなかった連続回数。patienceに達したら早期終了する。
    waiting = 0
    # 最良エポックの重みをCPU上へ独立コピーしたもの。学習中の更新から切り離す。
    best_state: dict[str, torch.Tensor] | None = None

    # 最大エポック数まで学習する。検証損失が改善しなければpatienceで途中終了する。
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        # 入力と正解を同じバッチで取り出し、バッチごとに勾配を初期化してパラメーターを更新する。
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
        # 一エポックの学習損失を観測行数で平均した値。
        train_loss = total_loss / max(seen, 1)

        model.eval()
        # 評価・推論では勾配を保存しない。パラメーター更新をせず、メモリ使用量も抑える。
        with torch.no_grad():
            val_logits = model(x_val_t)
            # 検証データでの損失。モデルの最良状態と早期終了を決める基準。
            val_loss = float(criterion(val_logits, y_val_t).item())
            # 勾配を記録せずに求めた学習データの3着内確率。AUCの監視に使う。
            train_prob = torch.sigmoid(model(x_train_t)).cpu().numpy()
            # 検証logitをsigmoidで0～1へ変換した確率。
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        train_auc = safe_auc(y_train, train_prob)
        val_auc = safe_auc(y_val, val_prob)
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
        # 進捗の通知先が渡された場合だけ通知する。CLIなど通知不要の利用でも同じ処理を使える。
        if progress:
            progress(epoch / config.epochs)

        # 微小な変動を改善と数えず、検証損失が十分に下がった時点の重みを独立コピーする。
        # 最終エポックの重みではなく、このbest_stateをテスト評価と保存に使用する。
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

    # 保存できる検証改善が一度もなければ、未選択の重みを成功モデルとして登録しない。
    if best_state is None:
        raise RuntimeError("学習済みモデルを保存できませんでした。")
    # テスト区分は最良エポックの選択には使わず、選択後のモデルを評価するために使う。
    model.load_state_dict(best_state)
    model.to(device).eval()
    # 評価・推論では勾配を保存しない。パラメーター更新をせず、メモリ使用量も抑える。
    with torch.no_grad():
        # 最良エポックのモデルによるテスト確率。最終指標の算出へ使う。
        test_prob = torch.sigmoid(model(torch.from_numpy(x_test).to(device))).cpu().numpy()
    # 確率0.5を閾値にした二値判定。確率の順位によるレース内上位3頭とは別の評価。
    test_pred = (test_prob >= 0.5).astype(int)
    metrics = {
        "model_run_id": model_run_id,
        # 再学習後も当時の損失の重みを正確に復元できるよう保存する。
        "pos_weight": negatives / positives,
        "calibration_file": "calibration.json",
        "target_column": TARGET_COLUMN,
        "task_name": TASK_NAME,
        "estimator_name": ESTIMATOR_NAME,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "validation_auc": float(history[best_epoch - 1]["validation_auc"]),
        "test_auc": safe_auc(y_test, test_prob),
        "test_log_loss": float(log_loss(y_test, np.clip(test_prob, 1e-6, 1 - 1e-6))),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "input_dim": int(x_train.shape[1]),
        "device": str(device),
        "features": model_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "training_dataset_manifest": dataset.manifest,
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
            "raw_top3_probability": test_prob,
            "race_date": test_df["race_date"].to_numpy(),
            "model_run_id": model_run_id,
        }
    ).to_parquet(model_dir / "test_predictions.parquet", index=False)
    json_dump(model_dir / "metrics.json", metrics)
    # 本番の既定動作は補正なし。sigmoid校正は別の学習外データで明示的に作成する。
    save_calibrations(model_dir / "calibration.json", [
        ProbabilityCalibration(model_run_id),
        ProbabilityCalibration(model_run_id, 'class_weight', {'pos_weight': negatives / positives}),
    ])
    save_manifest(
        model_dir,
        {
            "model_run_id": model_run_id,
            "task_name": TASK_NAME,
            "estimator_name": ESTIMATOR_NAME,
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "target_column": TARGET_COLUMN,
            "features": model_features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "training_dataset": dataset.manifest,
            "created_at": datetime.now(),
            "data_start_date": training["race_date"].min(),
            "data_end_date": training["race_date"].max(),
        },
    )

    save_model_run(
        {
            "model_run_id": model_run_id,
            "created_at": datetime.now(),
            "status": "completed",
            "target": TASK_NAME,
            "task_name": TASK_NAME,
            "estimator_name": ESTIMATOR_NAME,
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
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


# チェックポイントの構造でMLPを再作成し、同じ学習runの前処理器と指標を読み込む。
def load_model_bundle(model_dir: Path) -> tuple[MLP, ColumnTransformer, dict[str, object]]:
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    load_calibration(model_dir, metrics)  # モデルを読む前に補正ファイルの必須性と対応IDを検査。
    # model_dir: 重み・前処理器・指標が同じ学習runとして保存されたディレクトリ。
    # 重みと入力次元・隠れ層幅・Dropoutを持つ保存済みモデル情報。
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model = MLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # 学習runに対応する補完・標準化・カテゴリ符号化の前処理器。
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    return model, preprocessor, metrics


# 保存済みの列名を入力スキーマの正として扱う。列情報のない旧モデルは再学習を要求する。
def _features_for_saved_model(
    metrics: dict[str, object],
) -> list[str]:
    """Return the registered columns saved by feature-DB Top3 models."""
    # モデル指標に記録された学習時の特徴量一覧。
    saved_features = metrics.get("features")
    if isinstance(saved_features, list) and all(
        isinstance(column, str) for column in saved_features
    ):
        return saved_features
    raise ValueError(
        "このモデルには特徴量DBの列情報がありません。新方式で再学習してください。"
    )


# 保存済み数値・カテゴリ定義で欠損フラグと列順を復元してから、前処理器向けの型へそろえる。
def _prepare_saved_model_features(
    target: pd.DataFrame,
    metrics: dict[str, object],
) -> pd.DataFrame:
    # この学習runで採用する列の順序。推論時も同じ構成を使う。
    model_features = _features_for_saved_model(metrics)
    numeric = metrics.get("numeric_features")
    categorical = metrics.get("categorical_features")
    if not (
        isinstance(numeric, list)
        and isinstance(categorical, list)
        and all(isinstance(column, str) for column in [*numeric, *categorical])
    ):
        raise ValueError(
            "このモデルには特徴量DBの型情報がありません。新方式で再学習してください。"
        )
    # 保存済みスキーマに合わせて欠損フラグと列を整えた予想入力。
    aligned = align_saved_top3_features(target, model_features)
    return prepare_registered_feature_frame(aligned, numeric, categorical)



# 当日以降の結果を特徴量から除いて過去レースを予想し、実着順との比較表と保存先・集計を返す。
def predict_historical_race(
    historical_records: pd.DataFrame,
    race_id: str,
    model_dir: Path,
    *, calibration_method: str = "none",
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    """
    過去の指定レースを、対象レースより前の履歴だけで特徴量生成して予想し、
    実際の着順と比較する。

    注意:
        保存済みモデル自体が対象レースを学習に使用している場合、
        完全な未見データ検証にはならない。
    """
    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
    # model_dir: 重み・前処理器・指標が同じ学習runとして保存されたディレクトリ。
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
    featured = build_top3_prediction_features(
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
        _prepare_saved_model_features(target, metrics)
    ).astype("float32")
    # 評価・推論では勾配を保存しない。パラメーター更新をせず、メモリ使用量も抑える。
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
        "frame_number",
        "horse_id",
        "horse_name",
        "jockey_name",
        "odds",
        "popularity",
    ]
    result = target[result_columns].copy()
    result = attach_probabilities(result, probability, model_dir, metrics, calibration_method)
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

    # 予想実行のID。出力ファイル群と予想履歴をまとめて識別する。
    prediction_run_id = (
        f"backtest_{datetime.now():%Y%m%d_%H%M%S}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    output_dir = PATHS.predictions / prediction_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"historical_prediction_{race_id}.parquet"
    )
    result.to_parquet(output_path, index=False)
    save_comparison_outputs(result, output_dir, _storage.load_race_odds(result.race_id.astype(str).unique().tolist()), save_primary_plan=True)
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

# 共有の推論だけを行い、予想ファイルや予想履歴DBは更新しない。保存は呼出側が選択する。
def infer_race(
    historical_records: pd.DataFrame,
    upcoming_records: pd.DataFrame,
    race_id: str,
    model_dir: Path,
    *, feature_database: Path | None = None, calibration_method: str = "none",
) -> tuple[pd.DataFrame, dict]:
    """Run the shared inference path without persisting predictions or DB records."""
    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # upcoming_records: 予想対象の出馬表。結果がまだ分からない行として扱う。
    # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
    # model_dir: 重み・前処理器・指標が同じ学習runとして保存されたディレクトリ。
    # feature_database: 特徴量DBの指定先。省略可能な場合は既定のfeatures.duckdbを使う。
    model, preprocessor, metrics = load_model_bundle(model_dir)
    featured = (build_top3_prediction_features(historical_records, upcoming_records, feature_database=feature_database)
                if feature_database is not None else build_top3_prediction_features(historical_records, upcoming_records))
    target = featured[featured["race_id"].astype(str) == str(race_id)].copy()
    if target.empty:
        raise ValueError("指定したレースの出走データがありません。")
    matrix = preprocessor.transform(
        _prepare_saved_model_features(target, metrics)
    ).astype("float32")
    # 評価・推論では勾配を保存しない。パラメーター更新をせず、メモリ使用量も抑える。
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.from_numpy(matrix))).numpy()
    result = target[
        ["race_id", "race_date", "course_name", "race_number", "race_name", "horse_number", "frame_number", "horse_id", "horse_name", "jockey_name", "odds", "popularity"]
    ].copy()
    result = attach_probabilities(result, probability, model_dir, metrics, calibration_method)
    result["expected_value_index"] = result["top3_probability"] * pd.to_numeric(result["odds"], errors="coerce")
    result = result.sort_values("top3_probability", ascending=False).reset_index(drop=True)
    result.insert(0, "prediction_rank", np.arange(1, len(result) + 1))

    return result, metrics


# 1レースの推論結果をファイルと実行履歴へ保存し、予想表とParquetの保存先を返す。
def predict_race(
    historical_records: pd.DataFrame,
    upcoming_records: pd.DataFrame,
    race_id: str,
    model_dir: Path,
    *, calibration_method: str = "none",
) -> tuple[pd.DataFrame, Path]:
    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # upcoming_records: 予想対象の出馬表。結果がまだ分からない行として扱う。
    # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
    # model_dir: 重み・前処理器・指標が同じ学習runとして保存されたディレクトリ。
    result, metrics = infer_race(historical_records, upcoming_records, race_id, model_dir,
                                 calibration_method=calibration_method)

    # 予想実行のID。出力ファイル群と予想履歴をまとめて識別する。
    prediction_run_id = f"pred_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    output_dir = PATHS.predictions / prediction_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"prediction_{race_id}.parquet"
    result.to_parquet(output_path, index=False)
    save_comparison_outputs(result, output_dir, _storage.load_race_odds(result.race_id.astype(str).unique().tolist()), save_primary_plan=True)
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


# 同じ日の出走馬をまとめて推論し、日別の成果物と予想実行履歴を保存する。
def predict_race_date(
    historical_records: pd.DataFrame,
    upcoming_records: pd.DataFrame,
    race_date: object,
    model_dir: Path,
    *, calibration_method: str = "none",
) -> tuple[pd.DataFrame, Path]:
    """Predict every cached race on one date in a single inference run."""

    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # upcoming_records: 予想対象の出馬表。結果がまだ分からない行として扱う。
    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    # model_dir: 重み・前処理器・指標が同じ学習runとして保存されたディレクトリ。
    # 入力日付を変換・正規化した開催日。
    requested_date = pd.to_datetime(race_date, errors="coerce")
    # 解釈できない開催日は、空予想として返さず入力エラーにする。
    if pd.isna(requested_date):
        raise ValueError("予想対象の日付が正しくありません。")
    requested_date = requested_date.normalize()

    upcoming = upcoming_records.copy()
    upcoming["race_date"] = pd.to_datetime(upcoming["race_date"], errors="coerce")
    # 予想対象日に一致する出走馬だけを抽出した出馬表。
    date_targets = upcoming[
        upcoming["race_date"].dt.normalize().eq(requested_date)
    ].copy()
    # 指定日に対応する出馬表がなければ、他の日付を予想せず不足を明示する。
    if date_targets.empty:
        raise ValueError("指定日の出走データがありません。")

    model, preprocessor, metrics = load_model_bundle(model_dir)
    featured = build_top3_prediction_features(historical_records, date_targets)
    matrix = preprocessor.transform(
        _prepare_saved_model_features(featured, metrics)
    ).astype("float32")
    # 評価・推論では勾配を保存しない。パラメーター更新をせず、メモリ使用量も抑える。
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.from_numpy(matrix))).numpy()

    result_columns = [
        "race_id", "race_date", "course_name", "race_number", "race_name",
        "horse_number", "frame_number", "horse_id", "horse_name", "jockey_name", "odds",
        "popularity",
    ]
    result = featured[result_columns].copy()
    result = attach_probabilities(result, probability, model_dir, metrics, calibration_method)
    result["expected_value_index"] = (
        result["top3_probability"]
        * pd.to_numeric(result["odds"], errors="coerce")
    )
    result["prediction_rank"] = (
        result.groupby("race_id", sort=False)["top3_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    result = result.sort_values(
        ["course_name", "race_number", "race_id", "prediction_rank"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "race_prediction_count", result.groupby("race_id")["race_id"].transform("size"))

    # 予想実行のID。出力ファイル群と予想履歴をまとめて識別する。
    prediction_run_id = (
        f"date_pred_{requested_date:%Y%m%d}_{datetime.now():%H%M%S}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    output_dir = PATHS.predictions / prediction_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{requested_date:%Y-%m-%d}.parquet"
    result.to_parquet(output_path, index=False)
    save_comparison_outputs(result, output_dir, _storage.load_race_odds(result.race_id.astype(str).unique().tolist()), save_primary_plan=True)
    result.to_csv(
        output_dir / f"predictions_{requested_date:%Y-%m-%d}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_prediction_run(
        {
            "prediction_run_id": prediction_run_id,
            "created_at": datetime.now(),
            "model_run_id": metrics["model_run_id"],
            "race_id": f"date:{requested_date:%Y-%m-%d}",
            "row_count": len(result),
            "output_path": str(output_path),
        }
    )
    return result, output_path
