"""Optuna-based hyperparameter tuning for XGBoost."""

from __future__ import annotations

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error

from bus_tt.train.train_xgb import train_xgb
from bus_tt.tune.search_spaces import SPACE_REGISTRY
from bus_tt.utils.logging import get_logger

log = get_logger(__name__)


def tune_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_trials: int = 50,
    seed: int = 42,
) -> optuna.Study:
    space_fn = SPACE_REGISTRY["xgb"]

    def objective(trial: optuna.Trial) -> float:
        hp = space_fn(trial)
        model = train_xgb(X_train, y_train, X_val, y_val, params=hp, seed=seed)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    log.info(f"Best XGB trial: {study.best_trial.value:.6f} params={study.best_trial.params}")
    return study
