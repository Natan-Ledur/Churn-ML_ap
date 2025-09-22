# Customer Churn ML (End-to-End)

A production-style, portfolio-ready churn prediction project featuring:

- Modular code in `src/churn_ml`
- Typer-based CLI: train, eval, predict
- Unit tests with `pytest`
- EDA notebook (`notebooks/EDA.ipynb`) with tuning and interpretability
- FastAPI microservice for serving
- Packaging via `pyproject.toml`
- CI workflow (lint + tests)

## Project Structure

```
customer-churn-ml/
├── data/
├── models/
├── notebooks/
├── src/
│   └── churn_ml/
│       ├── __init__.py
│       ├── __main__.py
│       ├── api.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       └── model.py
├── tests/
├── README.md
├── README-PT.md
└── pyproject.toml
```

## Quickstart

1) Create and activate a Python 3.9+ virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install the project in editable mode (dev + notebook/API extras)

```powershell
pip install -e .\customer-churn-ml[dev,nb]
```

3) Run tests

```powershell
pytest -q
```

4) Train a model using the sample dataset

```powershell
python -m churn_ml train --data .\customer-churn-ml\data\sample_churn.csv --target Churn --out .\customer-churn-ml\models\model.joblib
```

5) Evaluate the model

```powershell
python -m churn_ml eval --data .\customer-churn-ml\data\sample_churn.csv --target Churn --model .\customer-churn-ml\models\model.joblib
```

6) Batch predictions

```powershell
python -m churn_ml predict --data .\customer-churn-ml\data\sample_churn.csv --model .\customer-churn-ml\models\model.joblib --out .\customer-churn-ml\predictions.csv
```

## EDA Notebook

Open `notebooks/EDA.ipynb` and run all cells. It includes:
- Data download/fallback, cleaning, and typing
- Pandera schema validation
- EDA (target distribution, correlations)
- Stratified train/val/test splits
- Feature engineering (simple derived features)
- Preprocessing with `ColumnTransformer`
- Baseline (DummyClassifier) + metrics (ROC AUC, PR AUC, log loss, F1)
- Models: Logistic Regression, RandomForest (XGBoost optional)
- Cross-validation and selection
- Hyperparameter tuning with Optuna
- Test evaluation with ROC/PR curves and confusion matrix
- Interpretability: permutation importance and SHAP (if available)
- Pipeline persistence with joblib + metadata
- Batch inference function and I/O contract
- Minimal FastAPI app for serving
- Reproducibility: seeds, versions, and `pip freeze` artifacts

If you need the notebook extras:

```powershell
pip install -e .\customer-churn-ml[nb]
```

## FastAPI Serving

1) Ensure a model exists at `models/model.joblib` (or set `CHURN_MODEL_PATH`).

2) Start the API using Uvicorn (factory pattern):

```powershell
uvicorn churn_ml.api:create_app --factory --host 0.0.0.0 --port 8000
```

3) Test request (PowerShell):

```powershell
$body = @{ records = @(@{ feat_num = 9; feat_cat = "a" }) } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $body
```

Alternatively, curl (Windows PowerShell escaping):

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" ^
  -d "{\"records\": [{\"feat_num\": 9, \"feat_cat\": \"a\"}]}"
```

## Design Highlights

- Clear separation of concerns (data, features, models, CLI, API)
- Reproducibility with seeds and recorded package versions
- Validations and tests for reliability
- Lightweight MLOps: CI pipeline and packaging

## Roadmap (Nice-to-haves)

- Dockerfile and devcontainer
- Experiment tracking (MLflow) or data versioning (DVC)
- More extensive test coverage (API contract, preprocessing edge cases)
- Probability calibration and PR AUC focus for imbalanced data
- Streamlit demo app for stakeholders

## License

MIT

---

# Deprecated README

This file is deprecated — the English documentation is now the main README.

See:
- README.md (English, primary)
- README-PT.md (Português)
