"""Tests for bus_tt.eval.metrics."""

import numpy as np

from bus_tt.eval.metrics import (
    mae,
    rmse,
    mape,
    r2,
    spearman,
    bias_mean,
    underestimation_rate,
    compute_all,
)


class TestMAE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == 1.0


class TestRMSE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 4.0])
        expected = np.sqrt((1 + 4) / 2)
        np.testing.assert_allclose(rmse(y_true, y_pred), expected)


class TestMAPE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mape(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([90.0, 180.0])
        assert abs(mape(y_true, y_pred) - 10.0) < 0.01


class TestR2:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2(y, y) == 1.0

    def test_mean_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.full_like(y, y.mean())
        np.testing.assert_allclose(r2(y, pred), 0.0, atol=1e-10)


class TestSpearman:
    def test_perfect_positive(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(spearman(y, y) - 1.0) < 1e-10

    def test_perfect_negative(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(spearman(y, -y) + 1.0) < 1e-10


class TestBiasMean:
    def test_no_bias(self):
        y = np.array([1.0, 2.0, 3.0])
        assert bias_mean(y, y) == 0.0

    def test_positive_bias(self):
        y_true = np.array([3.0, 4.0])
        y_pred = np.array([1.0, 2.0])
        assert bias_mean(y_true, y_pred) == 2.0


class TestUnderestimationRate:
    def test_all_under(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([5.0, 10.0])
        assert underestimation_rate(y_true, y_pred) == 100.0

    def test_none_under(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([15.0, 25.0])
        assert underestimation_rate(y_true, y_pred) == 0.0


class TestComputeAll:
    def test_returns_all_keys(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_all(y, y + 0.1)
        expected_keys = {"MAE", "RMSE", "MAPE(%)", "R2", "Spearman", "Bias_mean", "Underest(%)"}
        assert set(result.keys()) == expected_keys

    def test_values_are_float(self):
        y = np.random.rand(20) * 10 + 1
        result = compute_all(y, y + np.random.randn(20) * 0.5)
        for v in result.values():
            assert isinstance(v, float)
