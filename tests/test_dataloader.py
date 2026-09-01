"""Unit tests for dataloader.py - data fetching and formatting."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from dataloader import (
    fetch_klines,
    format_to_dataframe,
    SYMBOL,
    INTERVAL,
    LIMIT,
)


class TestDataloaderFetching:
    """Tests for Binance API data fetching."""

    @patch("dataloader.requests.get")
    def test_fetch_klines_success(self, mock_get):
        """Test successful kline data fetch."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [1609459200000, "100.0", "101.0", "99.0", "100.5", "1000.0", 1609459260000, "100500", 10, "500", "50000", "0"],
            [1609459260000, "100.5", "102.0", "100.0", "101.0", "1100.0", 1609459320000, "111000", 11, "550", "55000", "0"],
        ]
        mock_get.return_value = mock_response
        
        result = fetch_klines(SYMBOL, INTERVAL)
        
        assert len(result) == 2
        assert result[0][0] == 1609459200000  # First timestamp

    @patch("dataloader.requests.get")
    def test_fetch_klines_with_start_time(self, mock_get):
        """Test fetch_klines with start_time parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        start_time = 1609459200000
        fetch_klines(SYMBOL, INTERVAL, start_time=start_time)
        
        # Check that start_time was included in parameters
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["startTime"] == start_time

    @patch("dataloader.requests.get")
    def test_fetch_klines_api_error(self, mock_get):
        """Test error handling when API returns error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": -1121, "msg": "Invalid symbol"}
        mock_get.return_value = mock_response
        
        result = fetch_klines(SYMBOL, INTERVAL)
        
        # Should return empty list on error
        assert result == []

    @patch("dataloader.requests.get")
    def test_fetch_klines_empty_response(self, mock_get):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        result = fetch_klines(SYMBOL, INTERVAL)
        
        assert result == []


class TestDataloaderFormatting:
    """Tests for data formatting and DataFrame construction."""

    def test_format_to_dataframe_basic(self):
        """Test basic DataFrame formatting."""
        raw_data = [
            [1609459200000, "100.0", "101.0", "99.0", "100.5", "1000.0", 1609459260000, "100500", 10, "500", "50000", "0"],
            [1609459260000, "100.5", "102.0", "100.0", "101.0", "1100.0", 1609459320000, "111000", 11, "550", "55000", "0"],
        ]
        
        df = format_to_dataframe(raw_data)
        
        # Check structure
        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        
        # Check data types
        assert df["timestamp"].dtype in ["int64", "int32"]
        assert df["open"].dtype == float
        assert df["close"].dtype == float
        assert df["volume"].dtype == float

    def test_format_to_dataframe_values(self):
        """Test that values are correctly formatted."""
        raw_data = [
            [1609459200000, "100.0", "101.0", "99.0", "100.5", "1000.0", 1609459260000, "100500", 10, "500", "50000", "0"],
        ]
        
        df = format_to_dataframe(raw_data)
        
        # Check values
        assert df["open"].iloc[0] == 100.0
        assert df["high"].iloc[0] == 101.0
        assert df["low"].iloc[0] == 99.0
        assert df["close"].iloc[0] == 100.5
        assert df["volume"].iloc[0] == 1000.0
        # Timestamp should be in seconds, not milliseconds
        assert df["timestamp"].iloc[0] == 1609459200  # milliseconds / 1000

    def test_format_to_dataframe_sorting(self):
        """Test that DataFrame is sorted by timestamp."""
        raw_data = [
            [1609459260000, "100.5", "102.0", "100.0", "101.0", "1100.0", 1609459320000, "111000", 11, "550", "55000", "0"],
            [1609459200000, "100.0", "101.0", "99.0", "100.5", "1000.0", 1609459260000, "100500", 10, "500", "50000", "0"],
        ]
        
        df = format_to_dataframe(raw_data)
        
        # Check that timestamps are sorted
        assert (df["timestamp"].diff().dropna() >= 0).all()
        # First timestamp should be the earlier one
        assert df["timestamp"].iloc[0] == 1609459200

    def test_format_to_dataframe_empty(self):
        """Test formatting empty data."""
        raw_data = []
        df = format_to_dataframe(raw_data)
        
        assert len(df) == 0
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_format_to_dataframe_string_conversion(self):
        """Test that string prices are correctly converted to floats."""
        raw_data = [
            [1609459200000, "99.99", "100.50", "98.50", "99.75", "5000.5", 1609459260000, "499750", 50, "2500", "250000", "0"],
        ]
        
        df = format_to_dataframe(raw_data)
        
        assert df["open"].iloc[0] == pytest.approx(99.99)
        assert df["high"].iloc[0] == pytest.approx(100.50)
        assert df["volume"].iloc[0] == pytest.approx(5000.5)


class TestDataloaderIntegration:
    """Integration tests for dataloader."""

    @patch("dataloader.requests.get")
    @patch("dataloader.time.sleep")
    def test_download_historical_pagination(self, mock_sleep, mock_get):
        """Test that download handles pagination correctly."""
        # Create mock response data
        mock_response = MagicMock()
        
        # First request returns LIMIT items, second request returns fewer
        data_page1 = [
            [1609459200000 + i*60000, "100.0", "101.0", "99.0", "100.5", "1000.0", 
             1609459200000 + (i+1)*60000, "100500", 10, "500", "50000", "0"]
            for i in range(LIMIT)
        ]
        data_page2 = [
            [1609459200000 + (LIMIT+i)*60000, "100.0", "101.0", "99.0", "100.5", "1000.0", 
             1609459200000 + (LIMIT+i+1)*60000, "100500", 10, "500", "50000", "0"]
            for i in range(500)
        ]
        
        # Mock the requests to return data for two pages
        mock_get.return_value.json.side_effect = [data_page1, data_page2]
        
        from dataloader import download_historical
        result = download_historical(SYMBOL, INTERVAL, total_points=1500)
        
        # Should have fetched data across multiple calls
        assert len(result) == 1500


class TestDataloaderEdgeCases:
    """Tests for edge cases and error handling."""

    def test_format_to_dataframe_missing_columns(self):
        """Test that formatter handles data with expected columns."""
        # Valid API response format (12 columns)
        raw_data = [
            [1609459200000, "100.0", "101.0", "99.0", "100.5", "1000.0", 1609459260000, "100500", 10, "500", "50000", "0"],
        ]
        
        df = format_to_dataframe(raw_data)
        assert len(df) == 1

    def test_format_to_dataframe_invalid_prices(self):
        """Test handling of invalid price data."""
        raw_data = [
            [1609459200000, "invalid", "101.0", "99.0", "100.5", "1000.0", 1609459260000, "100500", 10, "500", "50000", "0"],
        ]
        
        # This should raise an error during type conversion
        with pytest.raises(ValueError):
            format_to_dataframe(raw_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
