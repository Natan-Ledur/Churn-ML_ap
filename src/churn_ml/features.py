from __future__ import annotations

from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    df: pd.DataFrame,
    cat_features: Optional[List[str]] = None,
    num_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    if cat_features is None:
        cat_features = [c for c in df.columns if df[c].dtype == "object"]
    if num_features is None:
        num_features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipe, cat_features),
            ("numerical", num_pipe, num_features),
        ]
    )
    return preprocessor
