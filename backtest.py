"""Run backtest on the trained model to evaluate trading performance.

This script:
1. Loads the trained model and normalization parameters
2. Runs a simulation on historical test data
3. Reports final balance, number of trades, average trade PnL, and win rate
4. Assumes fixed take profit and stop loss levels

Usage:
    python backtest.py
"""

import logging
import pandas as pd
from trade_model import (
    LSTMModel,
    add_features,
    create_labels,
    normalize_features,
    load_checkpoint,
    backtest,
)


def main():
    """Run backtest on test set."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load and preprocess data
    df = pd.read_csv("data.csv")
    df = add_features(df)
    df = create_labels(df)

    # Initialize model and load trained weights
    model = LSTMModel()
    means, stds = load_checkpoint(model)

    # Normalize features using training data statistics
    df = normalize_features(df, means, stds)
    
    # Split data: 80% train, 20% test
    split = int(len(df) * 0.8)
    test_df = df[split:]

    # Run backtest on test set
    logger.info("Backtesting loaded model on test data...")
    backtest(model, test_df)


if __name__ == "__main__":
    main()
