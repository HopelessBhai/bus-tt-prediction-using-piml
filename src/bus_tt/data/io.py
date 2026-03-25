from pathlib import Path

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
