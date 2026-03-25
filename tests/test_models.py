"""Tests for bus_tt.models (ANN, PINN, PhyLSTM, XGB builder)."""

import torch
import numpy as np

from bus_tt.models.ann import ANNModel
from bus_tt.models.pinn import PINN
from bus_tt.models.lstm import PhyLSTMModel
from bus_tt.models.xgb import build_xgb_model


class TestANNModel:
    def test_forward_shape(self):
        m = ANNModel(input_dim=6, hidden_dims=[128, 64])
        x = torch.randn(8, 6)
        out = m(x)
        assert out.shape == (8, 1)

    def test_single_hidden(self):
        m = ANNModel(input_dim=4, hidden_dims=[32])
        x = torch.randn(3, 4)
        assert m(x).shape == (3, 1)

    def test_multiple_hidden(self):
        m = ANNModel(input_dim=6, hidden_dims=[128, 64, 32])
        x = torch.randn(5, 6)
        assert m(x).shape == (5, 1)

    def test_dropout_param(self):
        m = ANNModel(input_dim=6, hidden_dims=[64], dropout=0.5)
        x = torch.randn(4, 6)
        assert m(x).shape == (4, 1)

    def test_gradient_flows(self):
        m = ANNModel(input_dim=6, hidden_dims=[32])
        x = torch.randn(4, 6, requires_grad=True)
        loss = m(x).sum()
        loss.backward()
        assert x.grad is not None


class TestPINN:
    def test_forward_shape(self):
        m = PINN(input_dim=6, output_dim=1, layer_dims=[128, 64])
        x = torch.randn(8, 6)
        assert m(x).shape == (8, 1)

    def test_different_output_dim(self):
        m = PINN(input_dim=6, output_dim=3, layer_dims=[64])
        x = torch.randn(4, 6)
        assert m(x).shape == (4, 3)

    def test_gradient_flows(self):
        m = PINN(input_dim=6, output_dim=1, layer_dims=[32])
        x = torch.randn(4, 6, requires_grad=True)
        loss = m(x).sum()
        loss.backward()
        assert x.grad is not None


class TestPhyLSTMModel:
    def test_forward_shape(self):
        m = PhyLSTMModel(hidden_dim=64, dropout=0.2)
        x_seq = torch.randn(8, 2, 1)
        x_ctx = torch.randn(8, 4)
        out = m(x_seq, x_ctx)
        assert out.shape == (8, 1)

    def test_different_hidden(self):
        m = PhyLSTMModel(hidden_dim=32, dropout=0.1)
        x_seq = torch.randn(4, 3, 1)
        x_ctx = torch.randn(4, 4)
        assert m(x_seq, x_ctx).shape == (4, 1)

    def test_gradient_flows(self):
        m = PhyLSTMModel(hidden_dim=32)
        x_seq = torch.randn(4, 2, 1, requires_grad=True)
        x_ctx = torch.randn(4, 4, requires_grad=True)
        loss = m(x_seq, x_ctx).sum()
        loss.backward()
        assert x_seq.grad is not None
        assert x_ctx.grad is not None

    def test_eval_mode(self):
        m = PhyLSTMModel(hidden_dim=32)
        m.eval()
        x_seq = torch.randn(2, 2, 1)
        x_ctx = torch.randn(2, 4)
        with torch.no_grad():
            out = m(x_seq, x_ctx)
        assert out.shape == (2, 1)


class TestBuildXgbModel:
    def test_returns_regressor(self):
        import xgboost as xgb
        model = build_xgb_model()
        assert isinstance(model, xgb.XGBRegressor)

    def test_default_params(self):
        model = build_xgb_model()
        assert model.n_estimators == 500
        assert model.max_depth == 6

    def test_custom_params(self):
        model = build_xgb_model(params={"n_estimators": 100, "max_depth": 3})
        assert model.n_estimators == 100
        assert model.max_depth == 3

    def test_seed_passed(self):
        model = build_xgb_model(seed=123)
        assert model.random_state == 123
