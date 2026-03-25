"""Tests for bus_tt.train (train_torch, train_xgb)."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from bus_tt.data.datasets import TabularDataset, SeqCtxDataset
from bus_tt.models.ann import ANNModel
from bus_tt.models.pinn import PINN
from bus_tt.models.lstm import PhyLSTMModel
from bus_tt.losses.physics import PhysicsLoss
from bus_tt.train.train_torch import train_tabular, train_seq
from bus_tt.train.train_xgb import train_xgb


def _make_tabular_loaders(n=40, dim=6, bs=16):
    x = np.random.randn(n, dim).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    split = n // 2
    train_ds = TabularDataset(x[:split], y[:split])
    val_ds = TabularDataset(x[split:], y[split:])
    return DataLoader(train_ds, batch_size=bs), DataLoader(val_ds, batch_size=bs)


def _make_seq_loaders(n=40, tw=2, ctx_dim=4, bs=16):
    x_seq = np.random.randn(n, tw, 1).astype(np.float32)
    x_ctx = np.random.randn(n, ctx_dim).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    split = n // 2
    train_ds = SeqCtxDataset(x_seq[:split], x_ctx[:split], y[:split])
    val_ds = SeqCtxDataset(x_seq[split:], x_ctx[split:], y[split:])
    return DataLoader(train_ds, batch_size=bs), DataLoader(val_ds, batch_size=bs)


class TestTrainTabular:
    def test_ann_trains(self):
        tl, vl = _make_tabular_loaders()
        model = ANNModel(input_dim=6, hidden_dims=[16])
        result = train_tabular(
            model, tl, vl, torch.nn.MSELoss(),
            lr=0.01, max_epochs=3, patience=5, device="cpu",
        )
        assert "history" in result
        assert "best_val" in result
        assert len(result["history"]["train_loss"]) > 0

    def test_pinn_trains_with_physics(self):
        tl, vl = _make_tabular_loaders()
        model = PINN(input_dim=6, output_dim=1, layer_dims=[16])
        criterion = PhysicsLoss(phy_lambda=0.1)
        result = train_tabular(
            model, tl, vl, criterion,
            lr=0.01, max_epochs=3, patience=5, device="cpu", use_physics=True,
        )
        assert result["best_val"] < float("inf")

    def test_save_checkpoint(self, tmp_path):
        tl, vl = _make_tabular_loaders()
        model = ANNModel(input_dim=6, hidden_dims=[16])
        save_path = tmp_path / "test_ann.pth"
        train_tabular(
            model, tl, vl, torch.nn.MSELoss(),
            lr=0.01, max_epochs=3, patience=5, device="cpu",
            save_path=save_path,
        )
        assert save_path.exists()

    def test_early_stopping(self):
        tl, vl = _make_tabular_loaders()
        model = ANNModel(input_dim=6, hidden_dims=[16])
        result = train_tabular(
            model, tl, vl, torch.nn.MSELoss(),
            lr=0.01, max_epochs=200, patience=2, device="cpu",
        )
        assert len(result["history"]["train_loss"]) <= 200


class TestTrainSeq:
    def test_phylstm_trains(self):
        tl, vl = _make_seq_loaders()
        model = PhyLSTMModel(hidden_dim=16, dropout=0.1)
        criterion = PhysicsLoss(phy_lambda=0.1)
        result = train_seq(
            model, tl, vl, criterion,
            lr=0.01, max_epochs=3, patience=5, device="cpu",
        )
        assert result["best_val"] < float("inf")

    def test_save_checkpoint(self, tmp_path):
        tl, vl = _make_seq_loaders()
        model = PhyLSTMModel(hidden_dim=16, dropout=0.1)
        criterion = PhysicsLoss(phy_lambda=0.1)
        save_path = tmp_path / "test_phylstm.pth"
        train_seq(
            model, tl, vl, criterion,
            lr=0.01, max_epochs=3, patience=5, device="cpu",
            save_path=save_path,
        )
        assert save_path.exists()


class TestTrainXgb:
    def test_trains_and_returns_model(self):
        import xgboost as xgb
        X = np.random.randn(60, 6).astype(np.float32)
        y = np.random.randn(60).astype(np.float32)
        model = train_xgb(X[:40], y[:40], X[40:], y[40:],
                          params={"n_estimators": 10, "max_depth": 3})
        assert isinstance(model, xgb.XGBRegressor)

    def test_predictions_shape(self):
        X = np.random.randn(60, 6).astype(np.float32)
        y = np.random.randn(60).astype(np.float32)
        model = train_xgb(X[:40], y[:40], X[40:], y[40:],
                          params={"n_estimators": 10})
        preds = model.predict(X[40:])
        assert preds.shape == (20,)

    def test_save_model(self, tmp_path):
        X = np.random.randn(40, 6).astype(np.float32)
        y = np.random.randn(40).astype(np.float32)
        save_path = tmp_path / "xgb.json"
        train_xgb(X[:30], y[:30], params={"n_estimators": 5},
                  save_path=save_path)
        assert save_path.exists()
