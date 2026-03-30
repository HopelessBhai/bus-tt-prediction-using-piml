import torch
import torch.nn as nn

V_F = 60 * 5 / 18  # free-flow speed in m/s


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.maximum(x, torch.tensor(1e-5, device=x.device)) / V_F)


def pde_residual(v_curr: torch.Tensor, v_prev: torch.Tensor, dx: float = 100.0) -> torch.Tensor:
    """Aw-Rascle velocity-form residual surrogate: dv/dt + dF(v)/dx.

    Uses:
        F(v) = (v^2 / 2) * (0.5 + ln(v / v_f))

    Reference for the velocity-form adaptation used here:
        Bharathi et al., Physica A 596 (2022), 127086.
        https://doi.org/10.1016/j.physa.2022.127086
    """
    v_curr_safe = torch.maximum(v_curr, torch.tensor(1e-5, device=v_curr.device))
    v_prev_safe = torch.maximum(v_prev, torch.tensor(1e-5, device=v_prev.device))

    dv_dt = (v_curr_safe - v_prev_safe) / (dx / v_curr_safe)

    Fv_prev = v_prev_safe ** 2 / 2.0 * (0.5 + _safe_log(v_prev_safe))
    Fv_curr = v_curr_safe ** 2 / 2.0 * (0.5 + _safe_log(v_curr_safe))
    dFv_dx = (Fv_curr - Fv_prev) / dx

    return (dv_dt + dFv_dx) ** 2


class PhysicsLoss(nn.Module):
    """MSE (or any data loss) + lambda * physics residual."""

    def __init__(self, phy_lambda: float = 0.14, data_loss: nn.Module | None = None):
        super().__init__()
        self.phy_lambda = phy_lambda
        self.data_loss = data_loss or nn.MSELoss()

    def forward(self, model_inputs: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        data = self.data_loss(y_pred, y_true)
        v_curr = y_pred[:, 0:1]
        v_prev = model_inputs[:, 0:1]
        phy = pde_residual(v_curr, v_prev).mean()
        return data + self.phy_lambda * phy
