"""Tests for bus_tt.data.io."""

import pandas as pd

from bus_tt.data.io import load_raw


class TestLoadRaw:
    def test_returns_dataframe(self, sample_csv):
        df = load_raw(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_date_column_is_datetime(self, sample_csv):
        df = load_raw(sample_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_all_columns_present(self, sample_csv):
        df = load_raw(sample_csv)
        assert "Date" in df.columns
        assert "Start time of the trip" in df.columns
        for i in range(1, 11):
            assert f"Section {i}" in df.columns

    def test_correct_row_count(self, sample_csv):
        df = load_raw(sample_csv)
        assert len(df) == 24  # 4 days * 6 time slots

    def test_loads_real_sample(self):
        df = load_raw("data/sample/sample_bus_travel_times.csv")
        assert len(df) == 10
        assert "Section 1" in df.columns
