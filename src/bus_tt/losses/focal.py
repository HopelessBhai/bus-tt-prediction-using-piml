import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss for regression: upweights hard (high-error) samples."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mae_loss = self.mae(outputs, targets)
        weight = torch.pow(torch.abs(outputs - targets), self.gamma)
        return (weight * mae_loss).mean()
