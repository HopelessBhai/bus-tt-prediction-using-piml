"""Tests for bus_tt.models.hybrid (predict_hybrid, latency_hybrid_per_sample)."""

import numpy as np
import torch

from bus_tt.models.lstm import PhyLSTMModel
from bus_tt.models.xgb import build_xgb_model
from bus_tt.models.hybrid import predict_hybrid, latency_hybrid_per_sample


def _setup():
    phylstm = PhyLSTMModel(hidden_dim=16, dropout=0.1)
    xgb_model = build_xgb_model(params={"n_estimators": 5})
    X_train = np.random.randn(30, 6).astype(np.float32)
    y_train = np.random.randn(30).astype(np.float32)
    xgb_model.fit(X_train, y_train)

    n = 10
    x_seq = np.random.randn(n, 2, 1).astype(np.float32)
    x_ctx = np.random.randn(n, 4).astype(np.float32)
    x_xgb = np.random.randn(n, 6).astype(np.float32)
    prev_tt = np.array([30, 70, 20, 80, 50, 90, 40, 65, 55, 10], dtype=np.float32)
    return phylstm, xgb_model, x_seq, x_ctx, x_xgb, prev_tt


class TestPredictHybrid:
    def test_output_shapes(self):
        phylstm, xgb_model, x_seq, x_ctx, x_xgb, prev_tt = _setup()
        preds, routes = predict_hybrid(
            x_seq, x_ctx, x_xgb, prev_tt, phylstm, xgb_model, torch.device("cpu")
        )
        assert preds.shape == (10,)
        assert routes.shape == (10,)

    def test_routes_match_threshold(self):
        phylstm, xgb_model, x_seq, x_ctx, x_xgb, prev_tt = _setup()
        _, routes = predict_hybrid(
            x_seq, x_ctx, x_xgb, prev_tt, phylstm, xgb_model,
            torch.device("cpu"), threshold=60.0,
        )
        expected = prev_tt > 60.0
        np.testing.assert_array_equal(routes, expected)

    def test_all_xgb_route(self):
        phylstm, xgb_model, x_seq, x_ctx, x_xgb, _ = _setup()
        high_tt = np.full(10, 200.0, dtype=np.float32)
        _, routes = predict_hybrid(
            x_seq, x_ctx, x_xgb, high_tt, phylstm, xgb_model, torch.device("cpu")
        )
        assert routes.all()

    def test_all_phylstm_route(self):
        phylstm, xgb_model, x_seq, x_ctx, x_xgb, _ = _setup()
        low_tt = np.full(10, 1.0, dtype=np.float32)
        _, routes = predict_hybrid(
            x_seq, x_ctx, x_xgb, low_tt, phylstm, xgb_model, torch.device("cpu")
        )
        assert not routes.any()


class TestLatencyHybridPerSample:
    def test_returns_dict(self):
        phylstm, xgb_model, x_seq, x_ctx, x_xgb, prev_tt = _setup()
        result = latency_hybrid_per_sample(
            x_seq, x_ctx, x_xgb, prev_tt, phylstm, xgb_model, torch.device("cpu")
        )
        assert "mean_ms" in result
        assert "xgb_route_pct" in result

    def test_n_samples_limit(self):
        phylstm, xgb_model, x_seq, x_ctx, x_xgb, prev_tt = _setup()
        result = latency_hybrid_per_sample(
            x_seq, x_ctx, x_xgb, prev_tt, phylstm, xgb_model,
            torch.device("cpu"), n_samples=3,
        )
        assert result["n_samples"] == 3
