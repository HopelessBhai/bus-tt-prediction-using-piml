"""Tests for bus_tt.eval (latency, compare)."""

import numpy as np
import pandas as pd
import torch

from bus_tt.eval.latency import (
    latency_torch_single_input,
    latency_torch_two_input,
    latency_xgb,
    summarize,
    count_params,
    build_latency_table,
)
from bus_tt.eval.compare import load_and_align, compare_non_peak
from bus_tt.models.ann import ANNModel
from bus_tt.models.lstm import PhyLSTMModel


class TestLatencyTorch:
    def test_single_input(self):
        m = ANNModel(input_dim=6, hidden_dims=[16])
        x = torch.randn(5, 6)
        times = latency_torch_single_input(m, x, torch.device("cpu"))
        assert len(times) == 5
        assert all(t > 0 for t in times)

    def test_two_input(self):
        m = PhyLSTMModel(hidden_dim=16)
        x_seq = torch.randn(5, 2, 1)
        x_ctx = torch.randn(5, 4)
        times = latency_torch_two_input(m, x_seq, x_ctx, torch.device("cpu"))
        assert len(times) == 5


class TestLatencyXgb:
    def test_xgb_latency(self):
        from bus_tt.models.xgb import build_xgb_model
        model = build_xgb_model(params={"n_estimators": 5})
        X = np.random.randn(20, 6).astype(np.float32)
        y = np.random.randn(20).astype(np.float32)
        model.fit(X, y)
        times = latency_xgb(model, X[:5])
        assert len(times) == 5


class TestSummarize:
    def test_keys(self):
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = summarize(times)
        assert "mean_ms" in s
        assert "std_ms" in s
        assert "median_ms" in s
        assert "p95_ms" in s
        assert "n_samples" in s
        assert s["n_samples"] == 5


class TestCountParams:
    def test_ann(self):
        m = ANNModel(input_dim=6, hidden_dims=[16])
        n = count_params(m)
        assert n > 0
        assert isinstance(n, int)


class TestBuildLatencyTable:
    def test_returns_dataframe(self):
        results = {
            "A": {"mean_ms": 1.0, "std_ms": 0.1, "median_ms": 1.0, "p95_ms": 1.2, "n_samples": 10},
            "B": {"mean_ms": 2.0, "std_ms": 0.2, "median_ms": 2.0, "p95_ms": 2.4, "n_samples": 10},
        }
        df = build_latency_table(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_with_param_counts(self):
        results = {"A": {"mean_ms": 1.0}}
        df = build_latency_table(results, param_counts={"A": 100})
        assert "param_count" in df.columns


class TestLoadAndAlign:
    def test_loads_and_aligns(self, tmp_path):
        dates = ["2013-10-01", "2013-10-02"]
        times = ["07:30:00", "08:00:00"]
        rows = []
        for d in dates:
            for t in times:
                row = {"Date": d, "Start time of the trip": t}
                for s in range(6, 11):
                    row[f"Section {s}"] = float(np.random.randint(10, 50))
                rows.append(row)

        true_df = pd.DataFrame(rows)
        pred_df = true_df.copy()
        for s in range(6, 11):
            pred_df[f"Section {s}"] = pred_df[f"Section {s}"] + 1.0

        true_path = tmp_path / "true.csv"
        pred_path = tmp_path / "pred.csv"
        true_df.to_csv(true_path, index=False)
        pred_df.to_csv(pred_path, index=False)

        y_true, y_preds, section_cols = load_and_align(
            str(true_path), {"model1": str(pred_path)}
        )
        assert y_true.shape[0] == 4
        assert "model1" in y_preds
        assert len(section_cols) > 0


class TestCompareNonPeak:
    def test_returns_dataframe(self):
        y_true = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 70.0]])
        y_preds = {"m1": y_true + 1, "m2": y_true + 2}
        df = compare_non_peak(y_true, y_preds, threshold=60.0)
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns
        assert len(df) == 2
