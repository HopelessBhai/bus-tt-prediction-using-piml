"""Generic PyTorch training loops for tabular (ANN/PINN) and sequential (PhyLSTM) models."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from bus_tt.utils.logging import get_logger

log = get_logger(__name__)


def _move(tensors: tuple, device: torch.device):
    return tuple(t.to(device, non_blocking=True) for t in tensors)


def train_tabular(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    patience: int = 15,
    device: torch.device | str = "cpu",
    save_path: str | Path | None = None,
    use_physics: bool = False,
) -> dict:
    """Train ANN or PINN. Returns dict with train/val loss history and best state."""
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    scaler = GradScaler()

    best_val, best_state, wait = np.inf, None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x, y = _move(batch, device)
            optimizer.zero_grad()
            with autocast():
                pred = model(x)
                if use_physics:
                    loss = criterion(x, y, pred)
                else:
                    loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        tr_loss = np.mean(losses)

        model.eval()
        val_losses = []
        mse_fn = nn.MSELoss()
        with torch.no_grad():
            for batch in val_loader:
                x, y = _move(batch, device)
                pred = model(x)
                val_losses.append(mse_fn(pred, y).item())
        vl = np.mean(val_losses)
        scheduler.step(vl)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info(f"Early stop at epoch {epoch}")
                break

        if epoch % 10 == 0:
            log.info(f"Epoch {epoch:>4d}  train={tr_loss:.6f}  val={vl:.6f}")

    if best_state and save_path:
        torch.save(best_state, save_path)
        log.info(f"Saved best model -> {save_path}")

    model.load_state_dict(best_state or model.state_dict())
    return {"history": history, "best_val": best_val, "model": model}


def train_seq(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    *,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 150,
    patience: int = 20,
    device: torch.device | str = "cpu",
    save_path: str | Path | None = None,
) -> dict:
    """Train PhyLSTM (or plain LSTM). DataLoader yields (x_seq, x_ctx, y)."""
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    scaler = GradScaler()

    best_val, best_state, wait = np.inf, None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for x_seq, x_ctx, y in train_loader:
            x_seq, x_ctx, y = x_seq.to(device), x_ctx.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(x_seq, x_ctx)
                loss = criterion(x_ctx, y, pred)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        tr_loss = np.mean(losses)

        model.eval()
        val_losses = []
        mse_fn = nn.MSELoss()
        with torch.no_grad():
            for x_seq, x_ctx, y in val_loader:
                x_seq, x_ctx, y = x_seq.to(device), x_ctx.to(device), y.to(device)
                val_losses.append(mse_fn(model(x_seq, x_ctx), y).item())
        vl = np.mean(val_losses)
        scheduler.step(vl)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info(f"Early stop at epoch {epoch}")
                break

        if epoch % 10 == 0:
            log.info(f"Epoch {epoch:>4d}  train={tr_loss:.6f}  val={vl:.6f}")

    if best_state and save_path:
        torch.save(best_state, save_path)
        log.info(f"Saved best model -> {save_path}")

    model.load_state_dict(best_state or model.state_dict())
    return {"history": history, "best_val": best_val, "model": model}
