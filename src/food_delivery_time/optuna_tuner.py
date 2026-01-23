from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from food_delivery_time.service import ID, TARGET, _prepare_Xy_train, save_artifact
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score


def _objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "random_state": 42,
        "n_jobs": 1,
    }

    model = RandomForestRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1
    )
    return scores.mean()  # higher is better because it is neg MAE


def tune_and_train(
    data_path: str | Path,
    target_column: str = TARGET,
    n_trials: int = 30,
    model_dir: str | Path = "models",
    model_name: str = "optuna-best-model",
) -> dict[str, Any]:
    """
    Run Optuna to find hyperparameters and train a RandomForest model on raw data.
    Saves the trained model artifact in model_dir/model_name.joblib.
    """
    df = pd.read_csv(data_path)
    X, y = _prepare_Xy_train(df)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: _objective(trial, X, y), n_trials=n_trials)

    best_params = study.best_params | {"random_state": 42, "n_jobs": 1}

    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X, y)

    mae_full = mean_absolute_error(y, best_model.predict(X))

    artifact = {
        "model": best_model,
        "feature_columns": X.columns.tolist(),
        "target": target_column,
        "id_col": ID,
    }
    save_artifact(model_dir, model_name, artifact)

    return {
        "best_params": best_params,
        "best_score_cv": study.best_value,
        "mae_full": float(mae_full),
        "study": study,
        "model_path": str(Path(model_dir) / f"{model_name}.joblib"),
    }
