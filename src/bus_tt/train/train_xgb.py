"""XGBoost training helper."""

from pathlib import Path

import numpy as np
import xgboost as xgb

from bus_tt.models.xgb import build_xgb_model
from bus_tt.utils.logging import get_logger

log = get_logger(__name__)


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> xgb.XGBRegressor:
    model = build_xgb_model(params, seed=seed)

    fit_kw: dict = {}
    if X_val is not None and y_val is not None:
        fit_kw["eval_set"] = [(X_val, y_val)]
        fit_kw["verbose"] = False

    model.fit(X_train, y_train, **fit_kw)
    log.info(f"XGB trained — n_estimators={model.n_estimators}, best_score={model.best_score if hasattr(model, 'best_score') else 'N/A'}")

    if save_path:
        model.save_model(str(save_path))
        log.info(f"Saved XGB model -> {save_path}")

    return model
