# Trade ML - LSTM-Based Cryptocurrency Trading Model

A machine learning trading system that uses Long Short-Term Memory (LSTM) neural networks to predict Bitcoin price movements and generate trading signals.

## Overview

This project combines technical analysis with deep learning to:
- **Predict trading signals** (long, short, neutral) based on historical price data
- **Backtest strategies** on historical data to evaluate performance
- **Generate live trading signals** from real-time market data via Binance API

## Features

- **Technical Feature Engineering**: Computes RSI, ATR, moving averages, volatility, and volume metrics
- **LSTM Neural Network**: Processes sequences of 30 candlesticks to predict market direction
- **Class Weighting**: Handles imbalanced training data for improved model performance
- **Backtesting Engine**: Simulates trading with configurable take profit and stop loss levels
- **Live Signal Streaming**: Real-time predictions from Binance market data
- **Model Persistence**: Saves trained weights and normalization parameters for reproducibility

## Project Structure

```
trade-ml/
├── dataloader.py       # Download historical data from Binance API
├── trade_model.py      # Core LSTM model and utilities
├── train.py            # Train the model
├── evaluate.py         # Test model performance on validation set
├── backtest.py         # Backtest trading strategy on historical data
├── live.py             # Stream live trading signals
├── data.csv            # Historical market data (OHLCV)
└── model.pt            # Trained model checkpoint
```

## Installation

### Requirements
- Python 3.8+
- PyTorch
- pandas
- numpy
- requests

### Setup

```bash
# Install dependencies
pip install torch pandas numpy requests

# Download historical data
python dataloader.py

# Train the model (requires data.csv)
python train.py

# Evaluate on test set
python evaluate.py

# Backtest on historical data
python backtest.py

# Stream live signals (requires trained model)
python live.py
```

## Configuration

Key parameters in `trade_model.py`:

```python
SEQ_LEN = 30              # Sequence length for LSTM input
BATCH_SIZE = 64           # Training batch size
EPOCHS = 10               # Number of training epochs
LR = 0.001                # Learning rate
THRESHOLD = 0.65          # Probability threshold for trading signals
TAKE_PROFIT = 0.01        # 1% take profit target
STOP_LOSS = 0.005         # 0.5% stop loss
FEE = 0.0004              # Trading fee per trade
```

## Usage

### 1. Download Data
```bash
python dataloader.py
```
Downloads 100,000 1-minute BTC/USDT candles from Binance.

### 2. Train Model
```bash
python train.py              # Train if no checkpoint exists
python train.py --retrain    # Force retrain even if checkpoint exists
```

### 3. Evaluate Performance
```bash
python evaluate.py
```
Displays precision, recall, F1-score, and confusion matrix on the test set.

### 4. Backtest Strategy
```bash
python backtest.py
```
Simulates trading on historical data with configured take profit and stop loss.

### 5. Generate Live Signals
```bash
python live.py
```
Fetches real-time data and streams trading signals (press Ctrl+C to stop).

## Model Details

**Architecture:**
- LSTM layer: 64 hidden units
- Fully connected layer: 3-class output (neutral, long, short)
- Softmax activation for probability distribution

**Features:**
- Log returns
- Volatility (10-period standard deviation)
- Volume z-score
- Moving average difference (10 vs 20-period)
- RSI (14-period)
- ATR (14-period)
- Volume change percentage

**Labels:**
- `1`: Long signal (bullish - buy)
- `-1`: Short signal (bearish - sell)
- `0`: Neutral (no clear signal)

## Workflow

```
Data Download → Feature Engineering → Label Creation → Normalization
         ↓
Model Training → Evaluate → Backtest → Live Trading
```

## Data Split

- **80%** training data
- **20%** test/validation data

## Notes

- The model uses class weighting to handle imbalanced label distribution
- Features are normalized using training set statistics
- Live signals use the same normalization parameters computed during training
- Backtesting assumes fixed fee structure and no slippage
- Real trading performance may differ from backtest results

## Disclaimer

This is an educational project. Cryptocurrency trading involves significant risk. Use at your own discretion and never trade with capital you can't afford to lose.
