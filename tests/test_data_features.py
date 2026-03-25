"""Tests for bus_tt.data.features."""

import numpy as np
import pandas as pd

from bus_tt.data.features import (
    parse_time_column,
    add_time_features,
    get_section_cols,
    build_speed_df,
    build_samples,
    compute_prev_tt,
)
from bus_tt.constants import SECTION_LENGTH_M, DROP_SECTIONS


class TestParseTimeColumn:
    def test_parses_hhmmss(self):
        df = pd.DataFrame({"Start time of the trip": ["07:30:00", "08:00:00"]})
        result = parse_time_column(df)
        assert result.dt.hour.tolist() == [7, 8]

    def test_parses_hhmm(self):
        df = pd.DataFrame({"Start time of the trip": ["07:30", "08:00"]})
        result = parse_time_column(df)
        assert result.dt.minute.tolist() == [30, 0]


class TestAddTimeFeatures:
    def test_adds_columns(self, sample_df):
        df = add_time_features(sample_df)
        assert "_trip_minutes" in df.columns
        assert "Rounded Time Encoded" in df.columns
        assert "Day Encoded" in df.columns

    def test_sorted_by_date_and_time(self, sample_df):
        df = add_time_features(sample_df)
        dates = df["Date"].tolist()
        mins = df["_trip_minutes"].tolist()
        for i in range(1, len(df)):
            assert (dates[i], mins[i]) >= (dates[i - 1], mins[i - 1])

    def test_day_encoded_range(self, sample_df):
        df = add_time_features(sample_df)
        assert df["Day Encoded"].min() >= 0
        assert df["Day Encoded"].max() <= 6

    def test_rounded_time_encoded_is_float32(self, sample_df):
        df = add_time_features(sample_df)
        assert df["Rounded Time Encoded"].dtype == np.float32


class TestGetSectionCols:
    def test_returns_sorted_section_cols(self, sample_df):
        cols = get_section_cols(sample_df)
        nums = [int(c.split()[1]) for c in cols]
        assert nums == sorted(nums)
        assert len(cols) == 10

    def test_drop_prefix(self, sample_df):
        cols = get_section_cols(sample_df, drop_prefix=DROP_SECTIONS)
        for d in DROP_SECTIONS:
            assert d not in cols
        assert len(cols) == 5  # 10 - 5 dropped


class TestBuildSpeedDf:
    def test_speed_is_positive(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        assert (speed[cols] > 0).all().all()

    def test_speed_formula(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        tt_col = cols[0]
        expected = SECTION_LENGTH_M / processed_df[tt_col].clip(lower=1e-3)
        np.testing.assert_allclose(speed[tt_col].values, expected.values)


class TestBuildSamples:
    def test_output_keys(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        samples = build_samples(processed_df, cols, speed[cols].to_numpy())
        for key in ("X_seq", "X_ctx", "X_xgb", "y_speed", "row_idx", "sec_pos"):
            assert key in samples

    def test_shapes_consistent(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        samples = build_samples(processed_df, cols, speed[cols].to_numpy())
        n = len(samples["y_speed"])
        assert samples["X_seq"].shape[0] == n
        assert samples["X_ctx"].shape[0] == n
        assert samples["X_xgb"].shape[0] == n
        assert samples["row_idx"].shape[0] == n

    def test_x_seq_shape(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        samples = build_samples(processed_df, cols, speed[cols].to_numpy(), time_window=2)
        assert samples["X_seq"].shape[1] == 2  # time_window
        assert samples["X_seq"].shape[2] == 1

    def test_x_xgb_dim(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        samples = build_samples(processed_df, cols, speed[cols].to_numpy())
        assert samples["X_xgb"].shape[1] == 6

    def test_y_speed_positive(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        samples = build_samples(processed_df, cols, speed[cols].to_numpy())
        assert (samples["y_speed"] > 0).all()

    def test_dtypes(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        samples = build_samples(processed_df, cols, speed[cols].to_numpy())
        assert samples["X_seq"].dtype == np.float32
        assert samples["X_ctx"].dtype == np.float32
        assert samples["X_xgb"].dtype == np.float32
        assert samples["y_speed"].dtype == np.float32
        assert samples["row_idx"].dtype == np.int64


class TestComputePrevTt:
    def test_returns_two_arrays(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        samples = build_samples(processed_df, cols, speed[cols].to_numpy())
        true_tt, prev_tt = compute_prev_tt(
            processed_df, cols, samples["row_idx"], samples["sec_pos"]
        )
        assert true_tt.shape == prev_tt.shape
        assert len(true_tt) == len(samples["row_idx"])

    def test_no_nans(self, processed_df):
        cols = get_section_cols(processed_df, drop_prefix=DROP_SECTIONS)
        speed = build_speed_df(processed_df, cols)
        samples = build_samples(processed_df, cols, speed[cols].to_numpy())
        _, prev_tt = compute_prev_tt(
            processed_df, cols, samples["row_idx"], samples["sec_pos"]
        )
        assert not np.isnan(prev_tt).any()
