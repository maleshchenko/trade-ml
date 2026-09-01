"""Download historical candlestick data from Binance API.

This module provides utilities to:
1. Fetch candlestick (kline) data from Binance REST API
2. Download large amounts of historical data with multiple requests
3. Format raw API responses into a pandas DataFrame
4. Save data to CSV for training

Usage:
    python dataloader.py
"""

import logging
import requests
import pandas as pd
import time

logger = logging.getLogger(__name__)

# Binance API configuration
BASE_URL = "https://api.binance.com/api/v3/klines"

# Trading symbol (Bitcoin vs USDT)
SYMBOL = "BTCUSDT"
# Time interval for candlesticks (1 minute)
INTERVAL = "1m"   # 1m, 5m, 15m, 1h, etc.
# Maximum number of data points per API request
LIMIT = 1000      # max per request

def fetch_klines(symbol, interval, start_time=None):
    """Fetch candlestick data from Binance API for a given time range.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval (e.g., '1m', '5m', '1h')
        start_time: Optional Unix timestamp in milliseconds to start from
        
    Returns:
        List of raw kline data, or empty list if error occurs
    """
    # Set up API request parameters
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": LIMIT
    }

    # Add start time if provided (to resume from a specific point)
    if start_time:
        params["startTime"] = start_time

    # Make GET request to Binance API
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    # Check for API errors
    if isinstance(data, dict) and "code" in data:
        logger.error(f"API Error: {data}")
        return []

    return data


def download_historical(symbol, interval, total_points=5000):
    """Download historical data by making multiple API requests.
    
    Handles pagination to fetch large amounts of data while respecting rate limits.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval (e.g., '1m', '5m', '1h')
        total_points: Total number of candles to download
        
    Returns:
        List of all downloaded kline data
    """
    all_data = []
    
    # Calculate how many API requests we need
    num_requests = (total_points // LIMIT) + 1
    
    # Start from ~100 days ago (enough for 100k 1-min candles)
    # Each 1-min candle is 60 seconds = 60000 milliseconds
    current_time = int(time.time() * 1000)
    start_time = current_time - (num_requests * LIMIT * 60 * 1000)  # 60 seconds per candle
    
    # Fetch data in batches
    for i in range(num_requests):
        if len(all_data) >= total_points:
            break
            
        # Fetch data for current time window
        data = fetch_klines(symbol, interval, start_time)

        if not data:
            logger.warning(f"No data returned at request {i+1}")
            break

        # Add data to collection
        all_data.extend(data)
        logger.info(f"Downloaded {len(all_data)} rows")

        # Move to next time window (start after last close time from this batch)
        start_time = data[-1][6] + 1

        # Sleep to avoid hitting Binance API rate limits
        time.sleep(0.2)

    # Return only the requested number of points
    return all_data[:total_points]


def format_to_dataframe(raw):
    """Convert raw Binance API kline data to a pandas DataFrame.
    
    Args:
        raw: List of raw kline data from Binance API
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    # Create DataFrame with all columns from API response
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ])

    # Select only the relevant OHLCV columns
    df = df[["open_time", "open", "high", "low", "close", "volume"]]

    # Rename columns for clarity
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    # Convert data types
    df["timestamp"] = (df["timestamp"] // 1000).astype(int)  # Convert milliseconds to seconds
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # Sort by timestamp and reset index
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def main():
    """Main function: download historical data and save to CSV."""
    # Download raw historical data from Binance
    raw = download_historical(SYMBOL, INTERVAL, total_points=100000)

    # Format data into a structured DataFrame
    df = format_to_dataframe(raw)

    # Save to CSV file for use in training
    df.to_csv("data.csv", index=False)

    # Print confirmation and data preview
    logger.info("\nSaved to data.csv")
    logger.info(f"\nData preview:\n{df.head()}")


if __name__ == "__main__":
    main()
