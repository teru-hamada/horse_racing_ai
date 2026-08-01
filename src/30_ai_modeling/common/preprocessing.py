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
