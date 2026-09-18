"""Run backtest on the trained model to evaluate trading performance.

This script:
1. Loads the trained model and normalization parameters
2. Runs a simulation on historical test data
3. Reports final balance, number of trades, average trade PnL, and win rate
4. Assumes fixed take profit and stop loss levels

Usage:
    python backtest.py
    python backtest.py --start-date 2024-01-01 --end-date 2024-01-31
"""

import argparse
import logging
import pandas as pd
from dataloader import INTERVAL, SYMBOL, download_date_range, format_to_dataframe
from trade_model import (
    LSTMModel,
    add_features,
    create_labels,
    normalize_features,
    load_checkpoint,
    backtest,
    SEQ_LEN,
)


def inclusive_end_date(value):
    """Treat a date-only end bound as the end of that calendar day."""
    timestamp = pd.Timestamp(value)
    if isinstance(value, str) and len(value) == 10:
        timestamp += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return timestamp


def filter_date_range(df, start_date=None, end_date=None):
    """Return rows whose timestamps fall within the requested date range."""
    if start_date is None and end_date is None:
        return df

    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        timestamps = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    else:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Data contains invalid timestamps")

    start = pd.Timestamp(start_date) if start_date else timestamps.min()
    end = inclusive_end_date(end_date) if end_date else timestamps.max()
    if start > end:
        raise ValueError("start date must be before or equal to end date")

    mask = timestamps.ge(start) & timestamps.le(end)
    filtered = df.loc[mask].copy()
    if len(filtered) <= SEQ_LEN:
        raise ValueError(f"date range must contain more than {SEQ_LEN} rows for backtesting")
    return filtered.reset_index(drop=True)


def main(start_date=None, end_date=None):
    """Run backtest on test set."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load raw data and fetch the requested window when it is missing locally.
    df = pd.read_csv("data.csv")
    if start_date or end_date:
        timestamps = pd.to_datetime(df["timestamp"], unit="s")
        start = pd.Timestamp(start_date) if start_date else timestamps.min()
        end = inclusive_end_date(end_date) if end_date else timestamps.max()
        if ((timestamps >= start) & (timestamps <= end)).sum() <= SEQ_LEN:
            logger.info("Requested data is missing locally; downloading it from Binance...")
            downloaded = download_date_range(SYMBOL, INTERVAL, start, end)
            if not downloaded:
                raise ValueError("Binance returned no data for the requested date range")
            downloaded_df = format_to_dataframe(downloaded)
            df = pd.concat([df, downloaded_df], ignore_index=True)
            df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
            df.to_csv("data.csv", index=False)

    # Add features and labels after merging any downloaded candles.
    df = add_features(df)
    df = create_labels(df)

    # Initialize model and load trained weights
    model = LSTMModel()
    means, stds = load_checkpoint(model)

    # Normalize features using training data statistics
    df = normalize_features(df, means, stds)
    
    if start_date or end_date:
        test_df = filter_date_range(df, start_date, end_date)
        logger.info("Backtesting date range from %s to %s", start_date or "start", end_date or "end")
    else:
        # Split data: 80% train, 20% test
        split = int(len(df) * 0.8)
        test_df = df[split:]

    # Run backtest on test set
    logger.info("Backtesting loaded model on test data...")
    backtest(model, test_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the trading model.")
    parser.add_argument("--start-date", help="Inclusive start date (YYYY-MM-DD or timestamp).")
    parser.add_argument("--end-date", help="Inclusive end date (YYYY-MM-DD or timestamp).")
    args = parser.parse_args()
    main(start_date=args.start_date, end_date=args.end_date)
