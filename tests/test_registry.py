"""Tests for bus_tt.train.registry (model + loss factory)."""

import torch.nn as nn

from bus_tt.train.registry import (
    build_model,
    build_loss,
    MODEL_REGISTRY,
    LOSS_REGISTRY,
)
from bus_tt.models.ann import ANNModel
from bus_tt.models.pinn import PINN
from bus_tt.models.lstm import PhyLSTMModel
from bus_tt.losses.physics import PhysicsLoss
from bus_tt.losses.focal import FocalLoss


class TestModelRegistry:
    def test_all_keys_present(self):
        assert set(MODEL_REGISTRY.keys()) == {"ann", "pinn", "phylstm"}

    def test_build_ann(self):
        m = build_model("ann", input_dim=6, hidden_dims=[64])
        assert isinstance(m, ANNModel)

    def test_build_pinn(self):
        m = build_model("pinn", input_dim=6, output_dim=1, layer_dims=[64])
        assert isinstance(m, PINN)

    def test_build_phylstm(self):
        m = build_model("phylstm", hidden_dim=32)
        assert isinstance(m, PhyLSTMModel)

    def test_unknown_model_raises(self):
        import pytest
        with pytest.raises(KeyError):
            build_model("unknown")


class TestLossRegistry:
    def test_all_keys_present(self):
        assert set(LOSS_REGISTRY.keys()) == {"mse", "physics", "focal"}

    def test_build_mse(self):
        loss = build_loss("mse")
        assert isinstance(loss, nn.MSELoss)

    def test_build_physics(self):
        loss = build_loss("physics", phy_lambda=0.2)
        assert isinstance(loss, PhysicsLoss)

    def test_build_focal(self):
        loss = build_loss("focal", gamma=1.5)
        assert isinstance(loss, FocalLoss)

    def test_unknown_loss_raises(self):
        import pytest
        with pytest.raises(KeyError):
            build_loss("unknown")
