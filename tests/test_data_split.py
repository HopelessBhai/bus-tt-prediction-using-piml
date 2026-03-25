"""Tests for bus_tt.data.split."""

import pandas as pd
import numpy as np

from bus_tt.data.split import train_test_mask
from bus_tt.constants import TEST_DATES


class TestTrainTestMask:
    def test_returns_two_series(self, processed_df):
        train, test = train_test_mask(processed_df)
        assert isinstance(train, pd.Series)
        assert isinstance(test, pd.Series)

    def test_masks_are_complementary(self, processed_df):
        train, test = train_test_mask(processed_df)
        assert (train | test).all()
        assert not (train & test).any()

    def test_test_dates_matched(self, processed_df):
        _, test = train_test_mask(processed_df)
        test_df = processed_df[test]
        test_strs = test_df["Date"].dt.strftime("%Y-%m-%d").unique()
        for d in test_strs:
            assert d in TEST_DATES

    def test_custom_test_dates(self, processed_df):
        custom = ["2013-10-01"]
        train, test = train_test_mask(processed_df, test_dates=custom)
        test_df = processed_df[test]
        assert (test_df["Date"].dt.strftime("%Y-%m-%d") == "2013-10-01").all()

    def test_all_train_when_no_test_dates_match(self, processed_df):
        train, test = train_test_mask(processed_df, test_dates=["2099-01-01"])
        assert train.all()
        assert not test.any()
