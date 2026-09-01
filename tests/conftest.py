"""Pytest configuration and shared fixtures for Trade ML tests."""

import pytest
import pandas as pd
import numpy as np
import torch
from trade_model import add_features, create_labels, normalize_features


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    return 42


@pytest.fixture
def sample_ohlcv_data(random_seed):
    """Create sample OHLCV data for testing.
    
    Returns a DataFrame with 200 rows of candlestick data.
    """
    dates = pd.date_range("2024-01-01", periods=200, freq="1min")
    close_prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close_prices * 0.98 + np.random.randn(200) * 0.1,
        "high": close_prices * 1.02 + np.random.rand(200) * 0.5,
        "low": close_prices * 0.98 - np.random.rand(200) * 0.5,
        "close": close_prices,
        "volume": np.random.rand(200) * 1000,
    })
    return df


@pytest.fixture
def sample_with_features(sample_ohlcv_data):
    """Create sample data with computed features."""
    df = add_features(sample_ohlcv_data)
    return df


@pytest.fixture
def sample_labeled_data(sample_ohlcv_data):
    """Create sample data with features and labels."""
    df = add_features(sample_ohlcv_data)
    df = create_labels(df)
    df = normalize_features(df)
    return df


@pytest.fixture
def trending_data(random_seed):
    """Create sample data with a clear uptrend."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1min")
    trend = np.linspace(0, 10, 500)
    close_prices = 100 + trend + np.random.randn(500) * 0.5
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close_prices,
        "high": close_prices + np.abs(np.random.randn(500)) * 0.3,
        "low": close_prices - np.abs(np.random.randn(500)) * 0.3,
        "close": close_prices,
        "volume": np.random.rand(500) * 2000 + 1000,
    })
    return df


@pytest.fixture
def volatile_data(random_seed):
    """Create sample data with high volatility."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1min")
    close_prices = 100 * np.cumprod(1 + np.random.randn(500) * 0.02)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close_prices,
        "high": close_prices * (1 + np.abs(np.random.randn(500)) * 0.01),
        "low": close_prices * (1 - np.abs(np.random.randn(500)) * 0.01),
        "close": close_prices,
        "volume": np.random.rand(500) * 5000,
    })
    return df


@pytest.fixture
def batch_tensors():
    """Create sample batch tensors for model testing."""
    from trade_model import SEQ_LEN, FEATURE_COLS
    batch_size = 4
    x = torch.randn(batch_size, SEQ_LEN, len(FEATURE_COLS))
    y = torch.randint(0, 3, (batch_size,))
    return x, y
