"""Stream live trading signals from real-time market data.

This script:
1. Loads the trained model and normalization parameters
2. Fetches latest candlestick data from Binance API
3. Computes technical features and generates trading signals
4. Streams signals periodically (default: every 60 seconds for 1-minute candles)

The script runs continuously until interrupted with Ctrl+C.

Usage:
    python live.py
"""

from trade_model import LSTMModel, load_checkpoint, stream_live_signals


def main():
    """Start live signal streaming."""
    # Initialize model and load trained weights
    model = LSTMModel()
    means, stds = load_checkpoint(model)
    
    # Stream live signals (runs until Ctrl+C)
    stream_live_signals(model, means, stds)


if __name__ == "__main__":
    main()
