"""Optuna-based hyperparameter tuning for PyTorch models (ANN, PINN, PhyLSTM)."""

from __future__ import annotations

import optuna
import torch
from torch.utils.data import DataLoader

from bus_tt.train.registry import build_model, build_loss
from bus_tt.train.train_torch import train_tabular, train_seq
from bus_tt.tune.search_spaces import SPACE_REGISTRY
from bus_tt.utils.logging import get_logger

log = get_logger(__name__)


def tune_tabular(
    model_name: str,
    train_loader_fn,
    val_loader_fn,
    *,
    input_dim: int,
    n_trials: int = 50,
    max_epochs: int = 60,
    device: str = "cpu",
    use_physics: bool = False,
) -> optuna.Study:
    """
    train_loader_fn(batch_size) -> DataLoader
    val_loader_fn(batch_size)   -> DataLoader
    """
    space_fn = SPACE_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:
        hp = space_fn(trial)
        bs = hp.pop("batch_size", 256)
        lr = hp.pop("lr")
        phy_lambda = hp.pop("phy_lambda", None)

        model = build_model(model_name, input_dim=input_dim, **hp)
        loss_name = "physics" if use_physics else "mse"
        loss_kw = {"phy_lambda": phy_lambda} if phy_lambda else {}
        criterion = build_loss(loss_name, **loss_kw)

        result = train_tabular(
            model,
            train_loader_fn(bs),
            val_loader_fn(bs),
            criterion,
            lr=lr,
            max_epochs=max_epochs,
            device=device,
            use_physics=use_physics,
        )
        return result["best_val"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    log.info(f"Best {model_name} trial: {study.best_trial.value:.6f} params={study.best_trial.params}")
    return study


def tune_seq(
    train_loader_fn,
    val_loader_fn,
    *,
    n_trials: int = 50,
    max_epochs: int = 80,
    device: str = "cpu",
) -> optuna.Study:
    space_fn = SPACE_REGISTRY["phylstm"]

    def objective(trial: optuna.Trial) -> float:
        hp = space_fn(trial)
        bs = hp.pop("batch_size", 256)
        lr = hp.pop("lr")
        phy_lambda = hp.pop("phy_lambda")
        wd = hp.pop("weight_decay", 1e-4)

        model = build_model("phylstm", **hp)
        criterion = build_loss("physics", phy_lambda=phy_lambda)

        result = train_seq(
            model,
            train_loader_fn(bs),
            val_loader_fn(bs),
            criterion,
            lr=lr,
            weight_decay=wd,
            max_epochs=max_epochs,
            device=device,
        )
        return result["best_val"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    log.info(f"Best phylstm trial: {study.best_trial.value:.6f} params={study.best_trial.params}")
    return study
