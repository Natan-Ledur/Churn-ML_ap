from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from .config import TrainConfig
from .data import train_test_split_df
from .features import build_preprocessor


def train_model(
    df: pd.DataFrame,
    config: TrainConfig,
    out_path: Optional[str | Path] = None,
) -> Pipeline:
    preprocessor = build_preprocessor(df.drop(columns=[config.target]), config.cat_features, config.num_features)
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])

    X_train, X_test, y_train, y_test = train_test_split_df(
        df, target=config.target, test_size=config.test_size, random_state=config.random_state
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=False)
    print("\nClassification Report:\n", report)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"config": asdict(config)}
        joblib.dump({"pipeline": pipe, "meta": meta}, out_path)
        print(f"Modelo salvo em: {out_path}")

    return pipe


def evaluate(
    df: pd.DataFrame, target: str, model_path: str | Path
) -> str:
    bundle = joblib.load(model_path)
    pipe: Pipeline = bundle["pipeline"]
    X = df.drop(columns=[target])
    y = df[target]
    y_pred = pipe.predict(X)
    rep = classification_report(y, y_pred)
    print(rep)
    return rep


def predict(
    df: pd.DataFrame, model_path: str | Path
) -> pd.Series:
    bundle = joblib.load(model_path)
    pipe: Pipeline = bundle["pipeline"]
    preds = pipe.predict(df)
    return pd.Series(preds, index=df.index)
