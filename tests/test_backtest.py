"""Tests for selecting backtest date ranges."""

import pandas as pd
import pytest

from backtest import filter_date_range


def test_filter_date_range_is_inclusive():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=80, freq="1min"),
        "close": range(80),
    })

    result = filter_date_range(df, "2024-01-01 00:10", "2024-01-01 00:50")

    assert len(result) == 41
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:10")
    assert result["timestamp"].iloc[-1] == pd.Timestamp("2024-01-01 00:50")


def test_filter_date_range_supports_unix_seconds_and_open_ended_bounds():
    df = pd.DataFrame({
        "timestamp": [1704067200 + index * 60 for index in range(80)],
    })

    result = filter_date_range(df, end_date="2024-01-01 00:50")

    assert len(result) == 51


def test_filter_date_range_rejects_reversed_or_short_ranges():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=40, freq="1min")})

    with pytest.raises(ValueError, match="start date"):
        filter_date_range(df, "2024-01-02", "2024-01-01")

    with pytest.raises(ValueError, match="more than"):
        filter_date_range(df, "2024-01-01 00:00", "2024-01-01 00:10")