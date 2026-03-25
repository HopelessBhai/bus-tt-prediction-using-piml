"""Multi-model comparison on aligned prediction CSVs."""

import numpy as np
import pandas as pd

from bus_tt.eval.metrics import mae, rmse, mape


def load_and_align(
    true_path: str,
    model_paths: dict[str, str],
    drop_sections: list[str] | None = None,
    lower_clip_pct: float | None = 5.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Load true + model CSVs, align by (Date, Start time), return arrays."""
    true_df = pd.read_csv(true_path)
    model_dfs = {name: pd.read_csv(p) for name, p in model_paths.items()}

    if drop_sections:
        true_df = true_df.drop(columns=drop_sections, errors="ignore")
        for name in model_dfs:
            model_dfs[name] = model_dfs[name].drop(columns=drop_sections, errors="ignore")

    section_cols = sorted(
        [c for c in true_df.columns if c.startswith("Section ") and all(c in d.columns for d in model_dfs.values())],
        key=lambda c: int(c.split()[1]),
    )

    if lower_clip_pct is not None:
        lower = np.percentile(true_df[section_cols].to_numpy(float), lower_clip_pct)
        true_df[section_cols] = true_df[section_cols].clip(lower=lower)
        for name in model_dfs:
            model_dfs[name][section_cols] = model_dfs[name][section_cols].clip(lower=lower)

    for d in [true_df, *model_dfs.values()]:
        d["key"] = d["Date"].astype(str) + "_" + d["Start time of the trip"].astype(str)

    common = set(true_df["key"])
    for d in model_dfs.values():
        common &= set(d["key"])

    true_df = true_df[true_df["key"].isin(common)].sort_values("key").reset_index(drop=True)
    for name in model_dfs:
        model_dfs[name] = model_dfs[name][model_dfs[name]["key"].isin(common)].sort_values("key").reset_index(drop=True)

    y_true = true_df[section_cols].to_numpy(float)
    y_preds = {name: d[section_cols].to_numpy(float) for name, d in model_dfs.items()}
    return y_true, y_preds, section_cols


def compare_non_peak(
    y_true: np.ndarray,
    y_preds: dict[str, np.ndarray],
    threshold: float = 60.0,
) -> pd.DataFrame:
    mask = y_true < threshold
    y_t = y_true[mask]
    rows = []
    for name, yp in y_preds.items():
        yp_m = yp[mask]
        rows.append({"Model": name, "MAE": mae(y_t, yp_m), "RMSE": rmse(y_t, yp_m), "MAPE(%)": mape(y_t, yp_m)})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
