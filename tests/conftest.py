"""Shared fixtures for bus_tt test suite."""

import numpy as np
import pandas as pd
import pytest
import torch

from bus_tt.constants import SECTION_LENGTH_M


@pytest.fixture
def sample_csv(tmp_path):
    """Write a minimal CSV that matches the real schema and return the path."""
    rows = []
    for day in ["2013-10-01", "2013-10-02", "2013-10-03", "2013-10-14"]:
        for t in ["07:30:00", "08:00:00", "08:30:00", "09:00:00", "09:30:00", "10:00:00"]:
            row = {"Date": day, "Start time of the trip": t}
            for s in range(1, 11):
                row[f"Section {s}"] = round(np.random.uniform(10, 50), 1)
            rows.append(row)
    df = pd.DataFrame(rows)
    p = tmp_path / "test_data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def sample_df(sample_csv):
    from bus_tt.data.io import load_raw
    return load_raw(sample_csv)


@pytest.fixture
def processed_df(sample_df):
    from bus_tt.data.features import add_time_features
    return add_time_features(sample_df)


@pytest.fixture
def device():
    return torch.device("cpu")
