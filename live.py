"""Stream live trading signals from real-time market data.

This script:
1. Loads the trained model and normalization parameters
2. Fetches latest candlestick data from Binance API
3. Computes technical features and generates trading signals
4. Streams signals periodically (default: every 60 seconds for 1-minute candles)

The script runs continuously until interrupted with Ctrl+C.

Usage:
    python live.py
    python live.py --no-sound
"""

import argparse

from trade_model import LSTMModel, load_checkpoint, stream_live_signals


def main():
    """Start live signal streaming."""
    parser = argparse.ArgumentParser(description="Stream live trading signals.")
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="disable sounds when the trading signal changes",
    )
    args = parser.parse_args()

    # Initialize model and load trained weights
    model = LSTMModel()
    means, stds = load_checkpoint(model)
    
    # Stream live signals (runs until Ctrl+C)
    stream_live_signals(model, means, stds, sound_enabled=not args.no_sound)


if __name__ == "__main__":
    main()
