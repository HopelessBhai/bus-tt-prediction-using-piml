"""Standard regression metrics."""

import numpy as np
from scipy.stats import spearmanr


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-12))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(spearmanr(y_true, y_pred).correlation)


def bias_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true - y_pred).mean())


def underestimation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_pred < y_true).mean() * 100)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Spearman": spearman(y_true, y_pred),
        "Bias_mean": bias_mean(y_true, y_pred),
        "Underest(%)": underestimation_rate(y_true, y_pred),
    }
