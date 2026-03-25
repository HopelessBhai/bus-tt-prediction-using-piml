#!/usr/bin/env python3
"""Train a single model from a YAML config.

Usage:
    python scripts/train.py --config configs/train/phylstm.yaml
"""

import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from bus_tt.constants import DROP_SECTIONS
from bus_tt.data.io import load_raw
from bus_tt.data.features import add_time_features, get_section_cols, build_speed_df, build_samples
from bus_tt.data.split import train_test_mask
from bus_tt.data.datasets import SeqCtxDataset, TabularDataset
from bus_tt.train.registry import build_model, build_loss
from bus_tt.train.train_torch import train_tabular, train_seq
from bus_tt.train.train_xgb import train_xgb
from bus_tt.utils.seed import set_seed
from bus_tt.utils.logging import get_logger

log = get_logger("train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    df = load_raw(cfg["data_path"])
    df = add_time_features(df)
    section_cols = get_section_cols(df, drop_prefix=DROP_SECTIONS)
    speed_df = build_speed_df(df, section_cols)
    samples = build_samples(df, section_cols, speed_df[section_cols].to_numpy())

    train_mask, test_mask = train_test_mask(df)
    row_idx = samples["row_idx"]
    tr = train_mask.to_numpy()[row_idx]

    model_type = cfg["model"]["type"]
    save_dir = Path(cfg.get("output_dir", "outputs/models"))
    save_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "xgb":
        model = train_xgb(
            samples["X_xgb"][tr],
            samples["y_speed"][tr],
            samples["X_xgb"][~tr],
            samples["y_speed"][~tr],
            params=cfg["model"].get("params"),
            seed=cfg.get("seed", 42),
            save_path=save_dir / "xgb.json",
        )
        log.info("XGBoost training complete.")

    elif model_type in ("ann", "pinn"):
        bs = cfg["training"].get("batch_size", 256)
        train_ds = TabularDataset(samples["X_xgb"][tr], samples["y_speed"][tr])
        val_ds = TabularDataset(samples["X_xgb"][~tr], samples["y_speed"][~tr])
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs)

        model = build_model(model_type, **cfg["model"]["params"])
        use_physics = model_type == "pinn"
        loss_name = "physics" if use_physics else "mse"
        loss_kw = {k: v for k, v in cfg.get("loss", {}).items() if k != "type"}
        criterion = build_loss(cfg.get("loss", {}).get("type", loss_name), **loss_kw)

        result = train_tabular(
            model, train_loader, val_loader, criterion,
            lr=cfg["training"]["lr"],
            max_epochs=cfg["training"].get("max_epochs", 100),
            patience=cfg["training"].get("patience", 15),
            device=device,
            save_path=save_dir / f"{model_type}.pth",
            use_physics=use_physics,
        )
        log.info(f"{model_type} best val loss: {result['best_val']:.6f}")

    elif model_type == "phylstm":
        bs = cfg["training"].get("batch_size", 256)
        train_ds = SeqCtxDataset(samples["X_seq"][tr], samples["X_ctx"][tr], samples["y_speed"][tr])
        val_ds = SeqCtxDataset(samples["X_seq"][~tr], samples["X_ctx"][~tr], samples["y_speed"][~tr])
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs)

        model = build_model(model_type, **cfg["model"]["params"])
        loss_kw = {k: v for k, v in cfg.get("loss", {}).items() if k != "type"}
        criterion = build_loss(cfg.get("loss", {}).get("type", "physics"), **loss_kw)

        result = train_seq(
            model, train_loader, val_loader, criterion,
            lr=cfg["training"]["lr"],
            max_epochs=cfg["training"].get("max_epochs", 150),
            patience=cfg["training"].get("patience", 20),
            device=device,
            save_path=save_dir / "phylstm.pth",
        )
        log.info(f"PhyLSTM best val loss: {result['best_val']:.6f}")

    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()
