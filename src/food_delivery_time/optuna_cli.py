from __future__ import annotations

from pathlib import Path

import typer
from food_delivery_time import optuna_tuner

app = typer.Typer(help="Optuna hyperparameter search for the food delivery model.")


@app.command("tune")
def tune(
    data_path: Path = typer.Argument(
        ..., help="Path to CSV with raw training data (must include target column)."
    ),
    target: str = typer.Option(
        optuna_tuner.TARGET, "--target", "-t", help="Target column name."
    ),
    n_trials: int = typer.Option(30, "--n-trials", "-n", help="Number of Optuna trials."),
    model_dir: Path = typer.Option(
        Path("models"), "--model-dir", "-d", help="Directory to save tuned model."
    ),
    model_name: str = typer.Option(
        "optuna-best-model",
        "--model-name",
        "-m",
        help="Filename stem for saved model (without extension).",
    ),
):
    """
    Run Optuna to tune hyperparameters and train the best model.
    Saves the trained artifact to model_dir/model_name.joblib.
    """
    result = optuna_tuner.tune_and_train(
        data_path=data_path,
        target_column=target,
        n_trials=n_trials,
        model_dir=model_dir,
        model_name=model_name,
    )

    typer.echo(f"Best CV score (neg MAE): {result['best_score_cv']:.4f}")
    typer.echo(f"Full-data MAE: {result['mae_full']:.4f}")
    typer.echo(f"Best params: {result['best_params']}")
    typer.echo(f"Model saved to: {result['model_path']}")


if __name__ == "__main__":
    app()
