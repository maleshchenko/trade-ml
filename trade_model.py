"""Trade ML Model

A machine learning model for cryptocurrency trading using LSTM neural networks.
Predicts market signals (long/short/neutral) based on technical features and
implements backtesting and live trading capabilities.
"""

import os
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================
# CONFIG
# =====================
# Configuration parameters for model training, trading, and data processing
# Sequence length for LSTM input
SEQ_LEN = 30
# Batch size for training
BATCH_SIZE = 64
# Number of training epochs
EPOCHS = 10
# Learning rate for optimizer
LR = 0.001
# Probability threshold for entering trades
THRESHOLD = 0.65
# Take profit percentage
TAKE_PROFIT = 0.01
# Stop loss percentage
STOP_LOSS = 0.005
# Trading fee per trade
FEE = 0.0004

# Live signal configuration
BASE_URL = "https://api.binance.com/api/v3/klines"
REALTIME_SYMBOL = "BTCUSDT"
REALTIME_INTERVAL = "1m"
REALTIME_UPDATES = 5
REALTIME_SLEEP = 60

# File path to save or load the trained model checkpoint
MODEL_PATH = "model.pt"

# Feature columns used by the model
FEATURE_COLS = [
    "return",
    "volatility",
    "volume_z",
    "ma_diff",
    "rsi",
    "atr",
    "volume_change",
]

# =====================
# FEATURE ENGINEERING
# =====================
# Functions to compute technical indicators and normalize features

def compute_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI) indicator."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df, period=14):
    """Calculate Average True Range (ATR) volatility indicator."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_features(df):
    """Compute technical features used by the model.
    
    Creates log returns, volatility, volume metrics, moving averages, RSI, and ATR.
    """
    df = df.copy()
    # Log returns for capturing percentage price changes
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    # Volatility of returns over 10 periods
    df["volatility"] = df["return"].rolling(10).std()
    # Standardized volume (z-score)
    df["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
        df["volume"].rolling(20).std() + 1e-9
    )
    # Moving averages and their difference (trend indicator)
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma_diff"] = df["ma10"] - df["ma20"]
    # Volume change percentage
    df["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0.0)
    # Technical indicators
    df["rsi"] = compute_rsi(df["close"], period=14)
    df["atr"] = compute_atr(df, period=14)
    df = df.dropna().reset_index(drop=True)
    return df


def normalize_features(df, means=None, stds=None):
    """Normalize features to zero mean and unit variance.
    
    Uses provided normalization parameters or computes them from data.
    """
    df = df.copy()
    if means is None or stds is None:
        means = {col: df[col].mean() for col in FEATURE_COLS}
        stds = {col: df[col].std() + 1e-9 for col in FEATURE_COLS}

    for col in FEATURE_COLS:
        df[col] = (df[col] - means[col]) / stds[col]

    return df


def compute_normalization_params(df):
    """Compute mean and std deviation for each feature."""
    means = {col: df[col].mean() for col in FEATURE_COLS}
    stds = {col: df[col].std() + 1e-9 for col in FEATURE_COLS}
    return means, stds


def save_checkpoint(model, means, stds, path=MODEL_PATH):
    """Save model weights and normalization parameters to disk."""
    checkpoint = {
        "model_state": model.state_dict(),
        "means": means,
        "stds": stds,
    }
    torch.save(checkpoint, path)
    print(f"Saved model checkpoint to {path}")


def load_checkpoint(model, path=MODEL_PATH):
    """Load model weights and normalization parameters from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    try:
        checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model checkpoint from {path}")
    return checkpoint["means"], checkpoint["stds"]

# =====================
# LABELING
# =====================
# Functions to create trading labels for supervised learning

def create_labels(df, horizon=10, target=0.005, stop=0.003):
    """Create labels based on future price movements.
    
    Labels are assigned as: 1 (long signal), -1 (short signal), 0 (neutral).
    A signal is generated if target profit can be reached before stop loss.
    """
    df = df.copy()
    # Look ahead to see price movement
    future_high = df["high"].rolling(horizon).max().shift(-horizon)
    future_low = df["low"].rolling(horizon).min().shift(-horizon)
    entry_price = df["close"]

    # Calculate potential upside and downside moves
    up_move = (future_high - entry_price) / entry_price
    down_move = (entry_price - future_low) / entry_price

    # Assign labels: buy if upside > target and downside < stop, vice versa for short
    df["label"] = 0
    df.loc[(up_move > target) & (down_move < stop), "label"] = 1
    df.loc[(down_move > target) & (up_move < stop), "label"] = -1

    df = df.dropna().reset_index(drop=True)
    return df

# =====================
# DATASET
# =====================
# PyTorch Dataset class for loading training data

class TradingDataset(Dataset):
    """Dataset for LSTM trading model.
    
    Each sample consists of SEQ_LEN time steps of features and a label.
    Labels are converted to 3 classes: 0=neutral, 1=long, 2=short.
    """
    def __init__(self, df):
        self.features = df[FEATURE_COLS].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.features) - SEQ_LEN

    def __getitem__(self, idx):
        # Extract a sequence of SEQ_LEN time steps
        x = self.features[idx : idx + SEQ_LEN]
        y = self.labels[idx + SEQ_LEN]
        # Convert labels: 0=neutral, 1=long, 2=short
        y_class = 0 if y == 0 else (1 if y == 1 else 2)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_class, dtype=torch.long)

# =====================
# MODEL
# =====================
# LSTM neural network for trading signal prediction

class LSTMModel(nn.Module):
    """LSTM model for predicting market signals.
    
    Architecture:
    - LSTM layer: processes sequences of technical features
    - Fully connected layer: maps LSTM output to 3 classes
    - Softmax: outputs probability distribution over classes
    """
    def __init__(self, input_size=len(FEATURE_COLS), hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # 3 classes: neutral, long, short
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM processes the entire sequence
        out, _ = self.lstm(x)
        # Use only the last output of the sequence
        out = out[:, -1, :]
        # Fully connected layer to predict class probabilities
        out = self.fc(out)
        return self.softmax(out)

# =====================
# TRAIN
# =====================
# Training functions and utilities

def compute_class_weights(labels):
    """Compute weights to handle class imbalance.
    
    Gives higher weight to underrepresented classes.
    """
    class_ids = np.array([0 if y == 0 else 1 if y == 1 else 2 for y in labels])
    counts = np.bincount(class_ids, minlength=3)
    weights = np.array([len(class_ids) / (count + 1e-9) for count in counts], dtype=np.float32)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(model, dataloader, class_weights=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


def label_to_class(labels):
    """Convert raw labels to class indices (0, 1, 2)."""
    return np.array([0 if y == 0 else 1 if y == 1 else 2 for y in labels], dtype=int)


def predict_dataset(model, df):
    """Generate predictions for an entire dataset.
    
    Returns true labels and predicted class indices.
    """
    model.eval()
    features = df[FEATURE_COLS].values
    y_true = label_to_class(df["label"].values[SEQ_LEN:])
    predictions = []

    with torch.no_grad():
        # Make predictions for each position in the dataset
        for i in range(SEQ_LEN, len(df)):
            x = torch.tensor(features[i - SEQ_LEN : i], dtype=torch.float32).unsqueeze(0)
            probs = model(x)[0].detach().numpy()
            # Take the class with highest probability
            predictions.append(int(np.argmax(probs)))

    return y_true, np.array(predictions)


def classification_report(y_true, y_pred):
    """Generate precision, recall, and F1 scores for each class.
    
    Returns confusion matrix and per-class metrics.
    """
    num_classes = 3
    labels = ["neutral", "long", "short"]
    # Build confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Calculate metrics for each class
    report = []
    for i in range(num_classes):
        tp = cm[i, i]  # True positives
        fp = cm[:, i].sum() - tp  # False positives
        fn = cm[i, :].sum() - tp  # False negatives
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        support = cm[i, :].sum()
        report.append((labels[i], precision, recall, f1, support))

    return cm, report

# =====================
# BACKTEST
# =====================
# Functions for historical performance evaluation

def backtest(model, df):
    """Run backtest on historical data and calculate returns.
    
    Simulates trading using model signals with configured take profit and stop loss.
    """
    model.eval()
    features = df[FEATURE_COLS].values
    prices = df["close"].values
    balance = 1000  # Starting balance
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0
    trades = []  # Track PnL of each closed trade

    # Iterate through each bar in the dataset
    for i in range(SEQ_LEN, len(df)):
        x = torch.tensor(features[i - SEQ_LEN : i], dtype=torch.float32).unsqueeze(0)
        probs = model(x)[0].detach().numpy()
        long_prob = probs[1]
        short_prob = probs[2]
        price = prices[i]

        # Enter new position if not already in a trade
        if position == 0:
            if long_prob > THRESHOLD and long_prob > short_prob:
                position = 1
                entry_price = price * (1 + FEE)  # Include entry fee
            elif short_prob > THRESHOLD and short_prob > long_prob:
                position = -1
                entry_price = price * (1 - FEE)
        else:
            # Exit position if take profit or stop loss is triggered
            if position == 1:
                change = (price - entry_price) / entry_price
                if change >= TAKE_PROFIT or change <= -STOP_LOSS:
                    exit_price = price * (1 - FEE)  # Include exit fee
                    pnl = (exit_price - entry_price) / entry_price
                    balance *= (1 + pnl)
                    trades.append(pnl)
                    position = 0
            else:  # Short position
                change = (entry_price - price) / entry_price
                if change >= TAKE_PROFIT or change <= -STOP_LOSS:
                    exit_price = price * (1 + FEE)
                    pnl = (entry_price - exit_price) / entry_price
                    balance *= (1 + pnl)
                    trades.append(pnl)
                    position = 0

    # Print backtest results
    print(f"Final balance: {balance:.2f}")
    print(f"Trades: {len(trades)}")
    if trades:
        print(f"Avg trade: {np.mean(trades):.4f}")
        print(f"Win rate: {np.mean([t > 0 for t in trades]):.2f}")

# =====================
# LIVE SIGNALS
# =====================
# Functions for real-time trading signal generation

def fetch_latest_klines(symbol, interval, limit=SEQ_LEN + 1):
    """Fetch candlestick data from Binance API."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    response = requests.get(BASE_URL, params=params, timeout=10)
    data = response.json()
    if isinstance(data, dict) and "code" in data:
        raise RuntimeError(f"Binance API error: {data}")
    return data


def format_klines(raw):
    """Parse and format Binance kline data into DataFrame."""
    df = pd.DataFrame(raw, columns=[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qav",
        "num_trades",
        "taker_base_vol",
        "taker_quote_vol",
        "ignore",
    ])
    # Keep only the columns we need
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    # Convert timestamp from milliseconds to seconds
    df["timestamp"] = (df["timestamp"] // 1000).astype(int)
    # Convert price and volume to float
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def normalize_live_features(df, means, stds):
    """Normalize live features using precomputed statistics."""
    return normalize_features(df, means, stds)


def decode_signal(probs):
    """Convert probability array to signal string."""
    idx = int(np.argmax(probs))
    return ["neutral", "long", "short"][idx]


def stream_live_signals(
    model,
    means,
    stds,
    symbol=REALTIME_SYMBOL,
    interval=REALTIME_INTERVAL,
    sleep_seconds=REALTIME_SLEEP,
):
    """Stream live trading signals from Binance market data.
    
    Fetches latest candles, computes features, and generates trading signals periodically.
    """
    print(f"Starting live signal stream for {symbol} on {interval} interval")
    print("Press Ctrl+C to stop.")
    update = 0
    while True:
        try:
            # Fetch latest market data from Binance
            raw = fetch_latest_klines(symbol, interval, limit=100)
            live_df = format_klines(raw)
            live_df = add_features(live_df)
            live_df = normalize_live_features(live_df, means, stds)

            if len(live_df) < SEQ_LEN:
                print("Not enough live data to generate a signal yet.")
                time.sleep(sleep_seconds)
                continue

            # Generate prediction using latest sequence
            x = torch.tensor(
                live_df[FEATURE_COLS].iloc[-SEQ_LEN:].values,
                dtype=torch.float32,
            ).unsqueeze(0)
            probs = model(x)[0].detach().numpy()
            signal = decode_signal(probs)
            last_price = live_df["close"].iloc[-1]
            update += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[{current_time}] [{update}] Price: {last_price:.2f}, Signal: {signal}, probs: {probs}")
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("\nLive signal stream interrupted.")
            break

    print("Live signal stream complete.")
