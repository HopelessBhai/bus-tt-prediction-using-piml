"""Per-sample latency benchmarking for all model types."""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def latency_torch_single_input(
    model: nn.Module, x: torch.Tensor, device: torch.device
) -> list[float]:
    """Per-sample latency (ms) for models that take a single tensor."""
    model.eval()
    times = []
    with torch.no_grad():
        for i in range(len(x)):
            _sync(device)
            t0 = time.perf_counter()
            model(x[i : i + 1].to(device))
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def latency_torch_two_input(
    model: nn.Module, x_seq: torch.Tensor, x_ctx: torch.Tensor, device: torch.device
) -> list[float]:
    """Per-sample latency (ms) for seq+ctx models (PhyLSTM)."""
    model.eval()
    times = []
    with torch.no_grad():
        for i in range(len(x_seq)):
            _sync(device)
            t0 = time.perf_counter()
            model(x_seq[i : i + 1].to(device), x_ctx[i : i + 1].to(device))
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def latency_xgb(model, x: np.ndarray) -> list[float]:
    times = []
    for i in range(len(x)):
        t0 = time.perf_counter()
        model.predict(x[i : i + 1])
        times.append((time.perf_counter() - t0) * 1000)
    return times


def summarize(times_ms: list[float]) -> dict[str, float]:
    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_samples": len(arr),
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_latency_table(results: dict[str, dict], param_counts: dict[str, int | str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df.index.name = "model"
    if param_counts:
        df["param_count"] = pd.Series(param_counts)
    return df
