import pandas as pd

from bus_tt.constants import TEST_DATES


def train_test_mask(df: pd.DataFrame, test_dates: list[str] | None = None) -> tuple[pd.Series, pd.Series]:
    dates = test_dates or TEST_DATES
    is_test = df["Date"].dt.strftime("%Y-%m-%d").isin(dates)
    return ~is_test, is_test
