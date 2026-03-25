"""Tests for bus_tt.losses (PhysicsLoss, FocalLoss, pde_residual)."""

import torch
import torch.nn as nn

from bus_tt.losses.physics import PhysicsLoss, pde_residual, V_F, _safe_log
from bus_tt.losses.focal import FocalLoss


class TestPdeResidual:
    def test_returns_tensor(self):
        v_curr = torch.tensor([10.0, 20.0])
        v_prev = torch.tensor([9.0, 18.0])
        r = pde_residual(v_curr, v_prev)
        assert isinstance(r, torch.Tensor)

    def test_non_negative(self):
        v_curr = torch.rand(10) * 30 + 1
        v_prev = torch.rand(10) * 30 + 1
        r = pde_residual(v_curr, v_prev)
        assert (r >= 0).all()

    def test_zero_when_equal(self):
        v = torch.tensor([10.0, 20.0])
        r = pde_residual(v, v)
        assert (r < 1e-3).all()


class TestSafeLog:
    def test_small_values_clamped(self):
        x = torch.tensor([0.0, -1.0, 1e-10])
        result = _safe_log(x)
        assert torch.isfinite(result).all()


class TestPhysicsLoss:
    def test_forward_returns_scalar(self):
        loss_fn = PhysicsLoss(phy_lambda=0.1)
        x = torch.randn(8, 6)
        y = torch.randn(8, 1)
        pred = torch.randn(8, 1)
        loss = loss_fn(x, y, pred)
        assert loss.dim() == 0

    def test_phy_lambda_effect(self):
        x = torch.rand(8, 6) + 0.1
        y = torch.rand(8, 1) + 0.1
        pred = torch.rand(8, 1) + 0.1
        loss_low = PhysicsLoss(phy_lambda=0.01)(x, y, pred)
        loss_high = PhysicsLoss(phy_lambda=10.0)(x, y, pred)
        assert loss_high > loss_low

    def test_custom_data_loss(self):
        loss_fn = PhysicsLoss(phy_lambda=0.1, data_loss=nn.L1Loss())
        x = torch.randn(4, 6)
        y = torch.randn(4, 1)
        pred = torch.randn(4, 1)
        loss = loss_fn(x, y, pred)
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        loss_fn = PhysicsLoss(phy_lambda=0.1)
        x = torch.rand(4, 6, requires_grad=True)
        y = torch.rand(4, 1)
        pred = torch.rand(4, 1, requires_grad=True)
        loss = loss_fn(x, y, pred)
        loss.backward()
        assert pred.grad is not None


class TestFocalLoss:
    def test_forward_returns_scalar(self):
        loss_fn = FocalLoss(gamma=2.0)
        out = torch.randn(8, 1)
        tgt = torch.randn(8, 1)
        loss = loss_fn(out, tgt)
        assert loss.dim() == 0

    def test_zero_when_perfect(self):
        loss_fn = FocalLoss(gamma=2.0)
        x = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(x, x)
        assert loss.item() == 0.0

    def test_non_negative(self):
        loss_fn = FocalLoss(gamma=2.0)
        out = torch.randn(10, 1)
        tgt = torch.randn(10, 1)
        assert loss_fn(out, tgt).item() >= 0

    def test_gamma_effect(self):
        out = torch.randn(10, 1)
        tgt = torch.randn(10, 1)
        l1 = FocalLoss(gamma=1.0)(out, tgt)
        l3 = FocalLoss(gamma=3.0)(out, tgt)
        assert l1 != l3
