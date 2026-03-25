"""Feature engineering: time encoding, speed conversion, sequence building."""

import numpy as np
import pandas as pd

from bus_tt.constants import SECTION_LENGTH_M, TIME_WINDOW


def parse_time_column(df: pd.DataFrame, col: str = "Start time of the trip") -> pd.Series:
    raw = df[col]
    try:
        return pd.to_datetime(raw, format="%H:%M:%S")
    except Exception:
        return pd.to_datetime(raw, format="%H:%M")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    time_parsed = parse_time_column(df)
    df["_trip_minutes"] = time_parsed.dt.hour * 60 + time_parsed.dt.minute
    df = df.sort_values(["Date", "_trip_minutes"]).reset_index(drop=True)

    rounded = ((df["_trip_minutes"] + 15) // 30) * 30 % (24 * 60)
    rmap = {v: i for i, v in enumerate(sorted(rounded.unique()))}
    df["Rounded Time Encoded"] = rounded.map(rmap).astype(np.float32)
    df["Day Encoded"] = df["Date"].dt.dayofweek.astype(np.float32)
    return df


def get_section_cols(df: pd.DataFrame, drop_prefix: list[str] | None = None) -> list[str]:
    cols = sorted(
        [c for c in df.columns if c.startswith("Section ")],
        key=lambda c: int(c.split()[1]),
    )
    if drop_prefix:
        cols = [c for c in cols if c not in drop_prefix]
    return cols


def build_speed_df(df: pd.DataFrame, section_cols: list[str]) -> pd.DataFrame:
    return SECTION_LENGTH_M / df[section_cols].clip(lower=1e-3)


def build_samples(
    df: pd.DataFrame,
    section_cols: list[str],
    speed_vals: np.ndarray,
    time_window: int = TIME_WINDOW,
) -> dict[str, np.ndarray]:
    """Build X_seq, X_ctx, X_xgb, y_speed, row/section indices."""
    rounded_arr = df["Rounded Time Encoded"].to_numpy(np.float32)
    day_arr = df["Day Encoded"].to_numpy(np.float32)

    n_rows, n_sec = speed_vals.shape
    X_seq, X_ctx, X_xgb, y_speed = [], [], [], []
    row_idx, sec_pos = [], []

    for si in range(1, n_sec):
        cur = speed_vals[:, si]
        prev = speed_vals[:, si - 1]
        sec_idx = si / float(n_sec)
        for t in range(time_window, n_rows):
            seq = cur[t - time_window : t].reshape(time_window, 1)
            ctx = np.array([prev[t], rounded_arr[t], day_arr[t], sec_idx * 100.0], dtype=np.float32)
            xgb_feat = np.array(
                [cur[t - 2], cur[t - 1], prev[t], rounded_arr[t], day_arr[t], sec_idx * 100.0],
                dtype=np.float32,
            )
            X_seq.append(seq)
            X_ctx.append(ctx)
            X_xgb.append(xgb_feat)
            y_speed.append(cur[t])
            row_idx.append(t)
            sec_pos.append(si)

    return {
        "X_seq": np.array(X_seq, dtype=np.float32),
        "X_ctx": np.array(X_ctx, dtype=np.float32),
        "X_xgb": np.array(X_xgb, dtype=np.float32),
        "y_speed": np.array(y_speed, dtype=np.float32),
        "row_idx": np.array(row_idx, dtype=np.int64),
        "sec_pos": np.array(sec_pos, dtype=np.int64),
    }


def compute_prev_tt(
    df: pd.DataFrame,
    section_cols: list[str],
    row_idx: np.ndarray,
    sec_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute true travel time and previous-trip travel time (causal)."""
    tt_vals = df[section_cols].to_numpy(np.float32)
    true_tt = tt_vals[row_idx, sec_pos]
    prev_tt = np.full_like(true_tt, np.nan)

    for _, grp in df.groupby(df["Date"].dt.date, sort=False):
        idx = grp.index.to_numpy()
        if len(idx) > 1:
            for k in range(1, len(idx)):
                mask = row_idx == idx[k]
                if mask.any():
                    prev_mask = row_idx == idx[k - 1]
                    if prev_mask.any():
                        prev_tt[mask] = tt_vals[idx[k - 1], sec_pos[mask]]

    prev_tt = np.nan_to_num(prev_tt, nan=np.nanmedian(true_tt))
    return true_tt, prev_tt
