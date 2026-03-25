#!/usr/bin/env python3
"""Evaluate a trained model on test data and print metrics.

Usage:
    python scripts/evaluate.py --config configs/train/phylstm.yaml --checkpoint outputs/models/phylstm.pth
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from bus_tt.constants import DROP_SECTIONS, SECTION_LENGTH_M
from bus_tt.data.io import load_raw
from bus_tt.data.features import add_time_features, get_section_cols, build_speed_df, build_samples
from bus_tt.data.split import train_test_mask
from bus_tt.data.datasets import SeqCtxDataset, TabularDataset
from bus_tt.train.registry import build_model
from bus_tt.eval.metrics import compute_all
from bus_tt.utils.seed import set_seed
from bus_tt.utils.logging import get_logger

log = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    df = load_raw(cfg["data_path"])
    df = add_time_features(df)
    section_cols = get_section_cols(df, drop_prefix=DROP_SECTIONS)
    speed_df = build_speed_df(df, section_cols)
    samples = build_samples(df, section_cols, speed_df[section_cols].to_numpy())

    _, test_mask = train_test_mask(df)
    row_idx = samples["row_idx"]
    te = test_mask.to_numpy()[row_idx]

    model_type = cfg["model"]["type"]

    if model_type in ("ann", "pinn"):
        model = build_model(model_type, **cfg["model"]["params"]).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

        x_test = torch.tensor(samples["X_xgb"][te], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = model(x_test).cpu().numpy().flatten()
        y_true_speed = samples["y_speed"][te]

    elif model_type == "phylstm":
        model = build_model(model_type, **cfg["model"]["params"]).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

        x_seq_t = torch.tensor(samples["X_seq"][te], dtype=torch.float32, device=device)
        x_ctx_t = torch.tensor(samples["X_ctx"][te], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = model(x_seq_t, x_ctx_t).cpu().numpy().flatten()
        y_true_speed = samples["y_speed"][te]

    elif model_type == "xgb":
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(args.checkpoint)
        preds = xgb_model.predict(samples["X_xgb"][te])
        y_true_speed = samples["y_speed"][te]

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    y_true_tt = SECTION_LENGTH_M / np.clip(y_true_speed, 1e-3, None)
    y_pred_tt = SECTION_LENGTH_M / np.clip(preds, 1e-3, None)

    metrics = compute_all(y_true_tt, y_pred_tt)
    log.info(f"Evaluation — {model_type}")
    for k, v in metrics.items():
        log.info(f"  {k:>15s}: {v:.4f}")


if __name__ == "__main__":
    main()
