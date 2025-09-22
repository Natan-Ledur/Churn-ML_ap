from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import TrainConfig, DEFAULT_TARGETS
from .data import load_data
from .model import evaluate as eval_model
from .model import predict as predict_model
from .model import train_model

app = typer.Typer(help="CLI para treinar, avaliar e prever churn")


@app.command()
def train(
    data: str = typer.Option(..., help="Caminho para CSV"),
    target: str = typer.Option(None, help="Nome da coluna alvo (default tenta detectar)"),
    out: str = typer.Option("models/model.joblib", help="Arquivo de saída do modelo"),
    test_size: float = typer.Option(0.2, help="Proporção de teste"),
    random_state: int = typer.Option(42, help="Semente aleatória"),
):
    df = load_data(data)
    if target is None:
        target = next((t for t in DEFAULT_TARGETS if t in df.columns), None)
        if target is None:
            raise typer.BadParameter("Não foi possível detectar a coluna alvo. Informe --target.")

    config = TrainConfig(target=target, test_size=test_size, random_state=random_state)
    print(f"[bold green]Treinando com target[/]: {target}")
    train_model(df, config, out_path=out)


@app.command("eval")
def eval_cmd(
    data: str = typer.Option(..., help="Caminho para CSV"),
    target: str = typer.Option(None, help="Nome da coluna alvo"),
    model: str = typer.Option(..., help="Caminho do modelo .joblib"),
):
    df = load_data(data)
    if target is None:
        target = next((t for t in DEFAULT_TARGETS if t in df.columns), None)
        if target is None:
            raise typer.BadParameter("Não foi possível detectar a coluna alvo. Informe --target.")
    print("[bold blue]Avaliando modelo...[/]")
    eval_model(df, target, model)


@app.command()
def predict(
    data: str = typer.Option(..., help="Caminho para CSV"),
    model: str = typer.Option(..., help="Caminho do modelo .joblib"),
    out: str = typer.Option("predictions.csv", help="Arquivo de saída"),
):
    df = load_data(data)
    print("[bold magenta]Gerando previsões...[/]")
    preds = predict_model(df, model)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False, header=["prediction"])
    print(f"[bold green]Predições salvas em[/]: {out_path}")


if __name__ == "__main__":
    app()
