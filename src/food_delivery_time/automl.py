from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pycaret.regression import RegressionExperiment
from sklearn import metrics

DEFAULT_TARGET = "Delivery_Time_min"


def train_best_model(
    data_path: str | Path,
    target_column: str = DEFAULT_TARGET,
    session_id: int | None = 42,
    model_name: str = "best-model",
    turbo: bool = False,
    fold: int = 5,
) -> dict[str, object]:
    """
    Train PyCaret AutoML on raw tabular data and persist the best model.

    Returns:
        dict with keys: model_path (str), leaderboard (pd.DataFrame),
        best_model (object), target_column (str).
    """
    data = pd.read_csv(data_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    exp = RegressionExperiment()
    exp.setup(
        data=data,
        target=target_column,
        session_id=session_id,
        fold=fold,
        verbose=False,
        log_experiment=False,
        remove_multicollinearity=False,
    )

    best_model = exp.compare_models(turbo=turbo)
    leaderboard = exp.pull().reset_index(drop=True)

    saved_model, model_path = exp.save_model(best_model, model_name)

    return {
        "model_path": str(model_path),
        "leaderboard": leaderboard,
        "best_model": saved_model,
        "target_column": target_column,
    }


def _load_model(model_path: str | Path) -> tuple[RegressionExperiment, object]:
    exp = RegressionExperiment()
    model = exp.load_model(str(model_path))
    return exp, model


def predict_with_model(
    model_path: str | Path,
    data_path: str | Path,
    target_column: Optional[str] = DEFAULT_TARGET,
) -> pd.DataFrame:
    """
    Run predictions with a saved PyCaret model on raw data.
    If target_column exists it is dropped before prediction.
    """
    df = pd.read_csv(data_path)
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])

    exp, model = _load_model(model_path)
    predictions = exp.predict_model(model, data=df)
    return predictions


def score_model(
    model_path: str | Path,
    data_path: str | Path,
    target_column: str = DEFAULT_TARGET,
) -> dict[str, float]:
    """
    Compute regression metrics for a saved model on a labeled dataset.
    """
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    y_true = df[target_column]
    features = df.drop(columns=[target_column])

    exp, model = _load_model(model_path)
    preds_df = exp.predict_model(model, data=features)

    if "prediction_label" not in preds_df.columns:
        raise ValueError("Prediction output missing 'prediction_label' column.")

    y_pred = preds_df["prediction_label"]

    return {
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "rmse": float(metrics.root_mean_squared_error(y_true, y_pred)),
        "r2": float(metrics.r2_score(y_true, y_pred)),
        "rows": int(len(df)),
    }
