# 登録された特徴量列の型を判定し、scikit-learn用の前処理器を構成する。
# 数値の欠損補完・標準化とカテゴリの補完・符号化を組み合わせる。
# 前処理器は学習データでfitし、検証・テスト・予想では同じ定義をtransformして用いる。
#
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 選択された入力列のdtypeから数値列とカテゴリ列を分け、固定の特徴量一覧に依存しないようにする。
def infer_feature_types(
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[list[str], list[str]]:
    """Split arbitrary registered features into numeric and categorical columns."""

    categorical = [
        column for column in features
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in features if column not in categorical]
    return numeric, categorical


# 数値・カテゴリを前処理器が受け取れる型へ統一し、予想時も学習時と同じ列を渡す。
def prepare_registered_feature_frame(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Normalize dynamically registered feature-set columns for sklearn."""

    features = [*numeric_features, *categorical_features]
    missing = set(features).difference(frame.columns)
    # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
    if missing:
        raise ValueError(f"Model input is missing feature columns: {sorted(missing)}")
    prepared = frame.loc[:, features].copy()
    for column in numeric_features:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in categorical_features:
        prepared[column] = prepared[column].astype(object)
        prepared[column] = prepared[column].where(prepared[column].notna(), np.nan)
    return prepared


# fit前の前処理器を構成する。平均やカテゴリ辞書の学習は呼出側が学習区分だけで実行する。
def make_registered_feature_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create preprocessing from feature metadata instead of hard-coded columns."""

    transformers = []
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                # 欠損は学習区分の中央値で補完し、その後で標準化する。
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                # 予想時の未知カテゴリで停止しないようignoreを指定し、低頻度カテゴリもまとめる。
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, min_frequency=2
                )),
            ]),
            categorical_features,
        ))
    # 利用する数値列もカテゴリ列もなければ、入力次元ゼロのモデルを作らない。
    if not transformers:
        raise ValueError("At least one model feature is required")
    return ColumnTransformer(transformers, remainder="drop")
