from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

TARGET = "Delivery_Time_min"
ID = "Order_ID"

CAT_COLS = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
NUM_COLS = ["Courier_Experience_yrs"]


def _artifact_path(model_dir: str | Path, name: str) -> Path:
    return Path(model_dir) / f"{name}.joblib"


def list_models(model_dir: str | Path) -> list[str]:
    p = Path(model_dir)
    if not p.exists():
        return []
    return sorted([f.stem for f in p.glob("*.joblib")])


def load_artifact(model_dir: str | Path, model_name: str) -> dict[str, Any]:
    path = _artifact_path(model_dir, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found in '{Path(model_dir).resolve()}'.")
    return joblib.load(path)


def save_artifact(model_dir: str | Path, model_name: str, artifact: dict[str, Any]) -> None:
    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, _artifact_path(p, model_name))


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist (basic guard)
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = None
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = None

    # Fill missing categoricals with a stable token
    df[CAT_COLS] = df[CAT_COLS].fillna("Unknown")

    # Fill missing numerics with median (or 0 if entire column missing)
    for col in NUM_COLS:
        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def _prepare_Xy_train(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df.columns:
        raise ValueError(f"Training input must include target column '{TARGET}'.")

    df = _clean_df(df)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any():
        raise ValueError("Target column contains non-numeric values or missing values.")

    X = df.drop(columns=[ID, TARGET], errors="ignore")
    X = pd.get_dummies(X)

    return X, y


def _prepare_X_predict(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET in df.columns:
        raise ValueError(f"Predict input must NOT include target column '{TARGET}'.")

    df = _clean_df(df)

    X = df.drop(columns=[ID], errors="ignore")
    X = pd.get_dummies(X)
    return X


def continue_train_from_rows(
    model_dir: str | Path,
    model_name: str,
    train_rows: list[dict],
    new_model_name: str,
) -> dict[str, float | int]:
    df = pd.DataFrame(train_rows)

    X, y = _prepare_Xy_train(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    try:
        base = load_artifact(model_dir, model_name)
        base_model: RandomForestRegressor = base["model"]
        model = RandomForestRegressor(
            random_state=42,
            n_estimators=base_model.n_estimators,
            max_depth=base_model.max_depth,
            min_samples_split=base_model.min_samples_split,
            min_samples_leaf=base_model.min_samples_leaf,
        )
    except FileNotFoundError:
        model = RandomForestRegressor(random_state=42)


    model.fit(X_train, y_train)

    # MAE on test split
    preds_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds_test)

    # MAE on full dataset
    preds_full = model.predict(X)
    mae_full = mean_absolute_error(y, preds_full)

    artifact = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "target": TARGET,
        "id_col": ID,
    }
    save_artifact(model_dir, new_model_name, artifact)

    return {
        "mae": float(mae),
        "mae_full": float(mae_full),
        "rows": int(len(df)),
        "features": int(len(X.columns)),
    }


def predict_from_rows(
    model_dir: str | Path,
    model_name: str,
    input_rows: list[dict],
) -> list[float]:
    df = pd.DataFrame(input_rows)
    X = _prepare_X_predict(df)

    artifact = load_artifact(model_dir, model_name)
    model: RandomForestRegressor = artifact["model"]
    cols: list[str] = artifact["feature_columns"]

    X_aligned = X.reindex(columns=cols, fill_value=0)
    preds = model.predict(X_aligned)
    return [float(p) for p in preds]
