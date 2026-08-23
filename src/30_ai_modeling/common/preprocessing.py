import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def prepare_feature_frame(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """pandasの欠損値をscikit-learnが扱える形に統一する。"""
    prepared = frame[features].copy()
    for column in CATEGORICAL_FEATURES:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype(object)
            prepared[column] = prepared[column].where(
                prepared[column].notna(),
                np.nan,
            )
    for column in NUMERIC_FEATURES:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column], errors="coerce"
            )
    return prepared


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")),
         ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [("imputer", SimpleImputer(
             strategy="constant",
             fill_value="__missing__",
         )),
         ("onehot", OneHotEncoder(
             handle_unknown="ignore",
             sparse_output=False,
             min_frequency=2,
         ))]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES),
         ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


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


def prepare_registered_feature_frame(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Normalize dynamically registered feature-set columns for sklearn."""

    features = [*numeric_features, *categorical_features]
    missing = set(features).difference(frame.columns)
    if missing:
        raise ValueError(f"Model input is missing feature columns: {sorted(missing)}")
    prepared = frame.loc[:, features].copy()
    for column in numeric_features:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in categorical_features:
        prepared[column] = prepared[column].astype(object)
        prepared[column] = prepared[column].where(prepared[column].notna(), np.nan)
    return prepared


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
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, min_frequency=2
                )),
            ]),
            categorical_features,
        ))
    if not transformers:
        raise ValueError("At least one model feature is required")
    return ColumnTransformer(transformers, remainder="drop")
