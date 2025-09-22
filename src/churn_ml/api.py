from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class Records(BaseModel):
    records: List[Dict[str, Any]]


DEFAULT_MODEL = os.getenv(
    "CHURN_MODEL_PATH",
    str((Path(__file__).resolve().parents[3] / "models" / "model.joblib").as_posix()),
)


def load_pipeline(path: str | Path):
    bundle = joblib.load(path)
    return bundle["pipeline"]


def create_app(model_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Churn ML API", version="0.1.0")
    pipe = load_pipeline(model_path or DEFAULT_MODEL)

    @app.post("/predict")
    def predict(payload: Records):
        df = pd.DataFrame(payload.records)
        preds = pipe.predict_proba(df)[:, 1]
        return {"predictions": preds.tolist()}

    return app
