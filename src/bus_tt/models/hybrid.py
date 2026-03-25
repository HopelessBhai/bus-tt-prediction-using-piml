"""Adaptive hybrid router: routes each sample to XGBoost or Phy-LSTM
based on the previous trip's travel time (temporal trigger)."""

import time

import numpy as np
import torch
import xgboost as xgb

from bus_tt.constants import TRIGGER_THRESHOLD_S


def predict_hybrid(
    x_seq: np.ndarray,
    x_ctx: np.ndarray,
    x_xgb: np.ndarray,
    prev_tt: np.ndarray,
    phylstm_model: torch.nn.Module,
    xgb_model: xgb.XGBRegressor,
    device: torch.device,
    threshold: float = TRIGGER_THRESHOLD_S,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        preds: array of predictions (n_samples,)
        routes: boolean mask — True where XGBoost was used, False for Phy-LSTM
    """
    n = len(prev_tt)
    trigger = prev_tt > threshold
    preds = np.zeros(n, dtype=np.float32)

    phylstm_model.eval()
    with torch.no_grad():
        for i in range(n):
            if trigger[i]:
                preds[i] = xgb_model.predict(x_xgb[i : i + 1])[0]
            else:
                seq_t = torch.tensor(x_seq[i : i + 1], dtype=torch.float32, device=device)
                ctx_t = torch.tensor(x_ctx[i : i + 1], dtype=torch.float32, device=device)
                preds[i] = phylstm_model(seq_t, ctx_t).item()
    return preds, trigger


def latency_hybrid_per_sample(
    x_seq: np.ndarray,
    x_ctx: np.ndarray,
    x_xgb: np.ndarray,
    prev_tt: np.ndarray,
    phylstm_model: torch.nn.Module,
    xgb_model: xgb.XGBRegressor,
    device: torch.device,
    threshold: float = TRIGGER_THRESHOLD_S,
    n_samples: int | None = None,
) -> dict:
    """Measure per-sample latency of the hybrid router."""
    if n_samples is not None:
        x_seq = x_seq[:n_samples]
        x_ctx = x_ctx[:n_samples]
        x_xgb = x_xgb[:n_samples]
        prev_tt = prev_tt[:n_samples]

    trigger = prev_tt > threshold
    phylstm_model.eval()
    times_ms: list[float] = []
    use_gpu = device.type == "cuda"

    with torch.no_grad():
        for i in range(len(prev_tt)):
            if use_gpu:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if trigger[i]:
                xgb_model.predict(x_xgb[i : i + 1])
            else:
                seq_t = torch.tensor(x_seq[i : i + 1], dtype=torch.float32, device=device)
                ctx_t = torch.tensor(x_ctx[i : i + 1], dtype=torch.float32, device=device)
                phylstm_model(seq_t, ctx_t)

            if use_gpu:
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times_ms)
    xgb_pct = trigger.mean() * 100
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_samples": len(arr),
        "xgb_route_pct": float(xgb_pct),
    }
