#!/usr/bin/env python3
"""Optuna hyper-parameter tuning from a YAML config.

Usage:
    python scripts/tune.py --config configs/tune/phylstm.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from bus_tt.constants import DROP_SECTIONS
from bus_tt.data.io import load_raw
from bus_tt.data.features import add_time_features, get_section_cols, build_speed_df, build_samples
from bus_tt.data.split import train_test_mask
from bus_tt.data.datasets import SeqCtxDataset, TabularDataset
from bus_tt.tune.optuna_torch import tune_tabular, tune_seq
from bus_tt.tune.optuna_xgb import tune_xgb
from bus_tt.utils.seed import set_seed
from bus_tt.utils.logging import get_logger

log = get_logger("tune")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))

    df = load_raw(cfg["data_path"])
    df = add_time_features(df)
    section_cols = get_section_cols(df, drop_prefix=DROP_SECTIONS)
    speed_df = build_speed_df(df, section_cols)
    samples = build_samples(df, section_cols, speed_df[section_cols].to_numpy())

    train_mask, _ = train_test_mask(df)
    row_idx = samples["row_idx"]
    tr = train_mask.to_numpy()[row_idx]

    model_type = cfg["model"]["type"]
    n_trials = cfg.get("n_trials", 50)
    out_dir = Path(cfg.get("output_dir", "outputs/tuning"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "xgb":
        study = tune_xgb(
            samples["X_xgb"][tr], samples["y_speed"][tr],
            samples["X_xgb"][~tr], samples["y_speed"][~tr],
            n_trials=n_trials,
        )
    elif model_type == "phylstm":
        def make_train(bs):
            return DataLoader(SeqCtxDataset(samples["X_seq"][tr], samples["X_ctx"][tr], samples["y_speed"][tr]), batch_size=bs, shuffle=True)
        def make_val(bs):
            return DataLoader(SeqCtxDataset(samples["X_seq"][~tr], samples["X_ctx"][~tr], samples["y_speed"][~tr]), batch_size=bs)
        study = tune_seq(make_train, make_val, n_trials=n_trials, device=device)
    elif model_type in ("ann", "pinn"):
        def make_train(bs):
            return DataLoader(TabularDataset(samples["X_xgb"][tr], samples["y_speed"][tr]), batch_size=bs, shuffle=True)
        def make_val(bs):
            return DataLoader(TabularDataset(samples["X_xgb"][~tr], samples["y_speed"][~tr]), batch_size=bs)
        study = tune_tabular(
            model_type, make_train, make_val,
            input_dim=samples["X_xgb"].shape[1],
            n_trials=n_trials, device=device,
            use_physics=(model_type == "pinn"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    result_path = out_dir / f"{model_type}_best.json"
    result_path.write_text(json.dumps(study.best_trial.params, indent=2))
    log.info(f"Best params saved -> {result_path}")


if __name__ == "__main__":
    main()
