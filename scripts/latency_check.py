#!/usr/bin/env python3
"""Per-sample latency benchmark for all models.

Usage:
    python scripts/latency_check.py --data path/to/data.csv
"""

import argparse

import numpy as np
import torch
import xgboost as xgb

from bus_tt.constants import DROP_SECTIONS, TRIGGER_THRESHOLD_S
from bus_tt.data.io import load_raw
from bus_tt.data.features import add_time_features, get_section_cols, build_speed_df, build_samples, compute_prev_tt
from bus_tt.data.split import train_test_mask
from bus_tt.models.ann import ANNModel
from bus_tt.models.pinn import PINN
from bus_tt.models.lstm import PhyLSTMModel
from bus_tt.models.xgb import build_xgb_model
from bus_tt.models.hybrid import latency_hybrid_per_sample
from bus_tt.eval.latency import (
    latency_torch_single_input,
    latency_torch_two_input,
    latency_xgb,
    summarize,
    count_params,
    build_latency_table,
)
from bus_tt.utils.seed import set_seed
from bus_tt.utils.logging import get_logger

log = get_logger("latency_check")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to raw CSV")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_raw(args.data)
    df = add_time_features(df)
    section_cols = get_section_cols(df, drop_prefix=DROP_SECTIONS)
    speed_df = build_speed_df(df, section_cols)
    samples = build_samples(df, section_cols, speed_df[section_cols].to_numpy())

    _, test_mask = train_test_mask(df)
    te = test_mask.to_numpy()[samples["row_idx"]]
    n = min(te.sum(), args.max_samples)

    X_seq_te = samples["X_seq"][te][:n]
    X_ctx_te = samples["X_ctx"][te][:n]
    X_xgb_te = samples["X_xgb"][te][:n]

    true_tt, prev_tt = compute_prev_tt(df, section_cols, samples["row_idx"][te][:n], samples["sec_pos"][te][:n])
    trigger = prev_tt > TRIGGER_THRESHOLD_S

    x_tab_t = torch.tensor(X_xgb_te, dtype=torch.float32, device=device)
    x_seq_t = torch.tensor(X_seq_te, dtype=torch.float32, device=device)
    x_ctx_t = torch.tensor(X_ctx_te, dtype=torch.float32, device=device)

    ann = ANNModel(input_dim=6, hidden_dims=[128, 64], dropout=0.1).to(device)
    pinn = PINN(input_dim=6, output_dim=1, layer_dims=[128, 64]).to(device)
    phylstm = PhyLSTMModel(hidden_dim=64, dropout=0.2).to(device)
    xgb_model = build_xgb_model()

    train_mask, _ = train_test_mask(df)
    tr = train_mask.to_numpy()[samples["row_idx"]]
    xgb_model.fit(samples["X_xgb"][tr][:5000], samples["y_speed"][tr][:5000])

    results = {
        "ANN": summarize(latency_torch_single_input(ann, x_tab_t, device)),
        "PINN": summarize(latency_torch_single_input(pinn, x_tab_t, device)),
        "Phy-LSTM": summarize(latency_torch_two_input(phylstm, x_seq_t, x_ctx_t, device)),
        "XGB": summarize(latency_xgb(xgb_model, X_xgb_te)),
        "Hybrid": latency_hybrid_per_sample(
            X_seq_te, X_ctx_te, X_xgb_te, prev_tt, phylstm, xgb_model, device
        ),
    }

    pcounts = {
        "ANN": count_params(ann),
        "PINN": count_params(pinn),
        "Phy-LSTM": count_params(phylstm),
        "XGB": f"{xgb_model.n_estimators} trees",
        "Hybrid": count_params(phylstm),
    }

    table = build_latency_table(results, pcounts)
    print(table.to_string())


if __name__ == "__main__":
    main()
