# Customer Churn ML

[Read this README in English](./README.md)

Um projeto completo de ciência de dados e ML para prever churn (saída de clientes), com:

- Código modular em `src/churn_ml`
- CLI com Typer (treinar, avaliar e prever)
- Testes com `pytest`
- Notebook de EDA em `notebooks/EDA.ipynb`
- Empacotado com `pyproject.toml`

## Estrutura

```
customer-churn-ml/
├── data/
├── models/
├── notebooks/
├── src/
│   └── churn_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       └── cli.py
├── tests/
├── README.md
└── pyproject.toml
```

## Como usar

1) Crie e ative um ambiente virtual Python 3.9+

```
python -m venv .venv
.venv\Scripts\activate
```

2) Instale o pacote (modo dev):

```
pip install -e .[dev]
```

3) Rode os testes

```
pytest -q
```

4) Use o dataset de exemplo `data/sample_churn.csv` e treine:

```
python -m churn_ml train --data data/sample_churn.csv --target Churn --out models/model.joblib
```

5) Avalie e gere métricas:

```
python -m churn_ml eval --data data/sample_churn.csv --target Churn --model models/model.joblib
```

6) Faça previsões em lote:

```
python -m churn_ml predict --data data/sample_churn.csv --model models/model.joblib --out predictions.csv
```

## API (serving com FastAPI)

1) Treine e gere um modelo em `models/model.joblib` (ou ajuste `CHURN_MODEL_PATH`).
2) Rode a API com Uvicorn:

```
uvicorn churn_ml.api:create_app --factory --host 0.0.0.0 --port 8000
```

3) Faça uma requisição de exemplo (PowerShell):

```powershell
$body = @{ records = @(@{ feat_num = 9; feat_cat = "a" }) } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $body
```

Ou com curl (escape no Windows PowerShell):

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" ^
  -d "{\"records\": [{\"feat_num\": 9, \"feat_cat\": \"a\"}]}"
```

## Notebook de EDA

Abra `notebooks/EDA.ipynb` e execute as células. Para os extras do notebook:

```
pip install -e .[nb]
```

## Licença

MIT