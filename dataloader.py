import requests
import pandas as pd
import time

# Base URL for Binance API klines endpoint
BASE_URL = "https://api.binance.com/api/v3/klines"

# Trading symbol (Bitcoin vs USDT)
SYMBOL = "BTCUSDT"
# Time interval for candlesticks (1 minute)
INTERVAL = "1m"   # 1m, 5m, 15m, 1h, etc.
# Maximum number of data points per API request
LIMIT = 1000      # max per request

# Function to fetch klines (candlestick data) from Binance API
def fetch_klines(symbol, interval, start_time=None):
    # Set up API request parameters
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": LIMIT
    }

    # Add start time if provided
    if start_time:
        params["startTime"] = start_time

    # Make GET request to API
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    # Check for API errors
    if isinstance(data, dict) and "code" in data:
        print(f"API Error: {data}")
        return []

    return data


# Function to download historical data by making multiple API requests
def download_historical(symbol, interval, total_points=5000):
    all_data = []
    
    # Calculate how many requests we need
    num_requests = (total_points // LIMIT) + 1
    
    # Start from ~100 days ago (enough for 100k 1-min candles)
    current_time = int(time.time() * 1000)
    start_time = current_time - (num_requests * LIMIT * 60 * 1000)  # 60 seconds per candle
    
    for i in range(num_requests):
        if len(all_data) >= total_points:
            break
            
        # Fetch data for current time window
        data = fetch_klines(symbol, interval, start_time)

        if not data:
            print(f"No data returned at request {i+1}")
            break

        # Add data to collection
        all_data.extend(data)
        print(f"Downloaded {len(all_data)} rows")

        # Move to next time window (start after last close time)
        start_time = data[-1][6] + 1

        # Sleep to avoid hitting rate limits
        time.sleep(0.2)  # avoid rate limits

    # Return only the requested number of points
    return all_data[:total_points]


# Function to convert raw API data to a pandas DataFrame
def format_to_dataframe(raw):
    # Create DataFrame with all columns from API response
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ])

    # Select only the relevant columns
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


# Main function to download and save historical data
def main():
    # Download raw historical data
    raw = download_historical(SYMBOL, INTERVAL, total_points=100000)

    # Format data into DataFrame
    df = format_to_dataframe(raw)

    # Save to CSV file
    df.to_csv("data.csv", index=False)

    # Print confirmation and preview
    print("\nSaved to data.csv")
    print(df.head())


if __name__ == "__main__":
    main()
