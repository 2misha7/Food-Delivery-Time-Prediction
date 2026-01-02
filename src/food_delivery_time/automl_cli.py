from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from food_delivery_time import automl

app = typer.Typer(help="PyCaret AutoML utilities for food delivery time prediction.")


@app.command()
def train(
    data_path: Path = typer.Argument(..., help="Path to CSV with raw training data."),
    target: str = typer.Option(
        automl.DEFAULT_TARGET, "--target", "-t", help="Target column name."
    ),
    model_name: str = typer.Option(
        "my-best-model", "--model-name", "-m", help="Base filename for the saved model."
    ),
    session_id: Optional[int] = typer.Option(
        42, "--session-id", help="Random seed for reproducibility."
    ),
    turbo: bool = typer.Option(
        False, "--turbo", help="Enable PyCaret turbo mode for faster model search."
    ),
    fold: int = typer.Option(5, "--fold", help="Cross-validation folds."),
    leaderboard_out: Optional[Path] = typer.Option(
        None, "--leaderboard-out", help="Optional path to write leaderboard CSV."
    ),
):
    """Train AutoML and persist the best model."""
    result = automl.train_best_model(
        data_path,
        target_column=target,
        session_id=session_id,
        model_name=model_name,
        turbo=turbo,
        fold=fold,
    )

    typer.echo(f"Saved best model to: {result['model_path']}")
    leaderboard: pd.DataFrame = result["leaderboard"]  # type: ignore[index]
    if leaderboard_out:
        leaderboard.to_csv(leaderboard_out, index=False)
        typer.echo(f"Leaderboard written to: {leaderboard_out}")
    else:
        typer.echo("Top models:")
        typer.echo(leaderboard.head())


@app.command()
def predict(
    model_path: Path = typer.Argument(..., help="Path to saved PyCaret model."),
    data_path: Path = typer.Argument(..., help="Path to CSV with raw input data."),
    output_path: Path = typer.Option(
        Path("predictions.csv"), "--output", "-o", help="Where to save predictions CSV."
    ),
    target: Optional[str] = typer.Option(
        automl.DEFAULT_TARGET,
        "--target",
        "-t",
        help="Optional target column to drop if present.",
    ),
):
    """Generate predictions with a saved model."""
    predictions = automl.predict_with_model(
        model_path=model_path,
        data_path=data_path,
        target_column=target,
    )
    predictions.to_csv(output_path, index=False)
    typer.echo(f"Wrote predictions to: {output_path}")


@app.command()
def score(
    model_path: Path = typer.Argument(..., help="Path to saved PyCaret model."),
    data_path: Path = typer.Argument(..., help="Path to labeled CSV."),
    target: str = typer.Option(
        automl.DEFAULT_TARGET, "--target", "-t", help="Target column name."
    ),
):
    """Compute regression metrics for a saved model on labeled data."""
    metrics = automl.score_model(
        model_path=model_path, data_path=data_path, target_column=target
    )
    typer.echo(metrics)


if __name__ == "__main__":
    app()
