"""Model + loss factory for config-driven training."""

import torch.nn as nn

from bus_tt.models.ann import ANNModel
from bus_tt.models.pinn import PINN
from bus_tt.models.lstm import PhyLSTMModel
from bus_tt.losses.physics import PhysicsLoss
from bus_tt.losses.focal import FocalLoss

MODEL_REGISTRY: dict[str, type] = {
    "ann": ANNModel,
    "pinn": PINN,
    "phylstm": PhyLSTMModel,
}

LOSS_REGISTRY: dict[str, type] = {
    "mse": nn.MSELoss,
    "physics": PhysicsLoss,
    "focal": FocalLoss,
}


def build_model(name: str, **kwargs) -> nn.Module:
    cls = MODEL_REGISTRY[name]
    return cls(**kwargs)


def build_loss(name: str, **kwargs) -> nn.Module:
    cls = LOSS_REGISTRY[name]
    return cls(**kwargs)
