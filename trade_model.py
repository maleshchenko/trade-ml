import os
import requests
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================
# CONFIG
# =====================
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

# =====================
# FEATURE ENGINEERING
# =====================

def add_features(df):
    df = df.copy()
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["return"].rolling(10).std()
    df["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
        df["volume"].rolling(20).std() + 1e-9
    )
    df = df.dropna().reset_index(drop=True)
    return df


def normalize_features(df, means=None, stds=None):
    cols = ["return", "volatility", "volume_z"]
    df = df.copy()

    if means is None or stds is None:
        for col in cols:
            mean = df[col].mean()
            std = df[col].std() + 1e-9
            df[col] = (df[col] - mean) / std
    else:
        for col in cols:
            df[col] = (df[col] - means[col]) / stds[col]

    return df


def compute_normalization_params(df):
    cols = ["return", "volatility", "volume_z"]
    means = {col: df[col].mean() for col in cols}
    stds = {col: df[col].std() + 1e-9 for col in cols}
    return means, stds


def save_checkpoint(model, means, stds, path=MODEL_PATH):
    checkpoint = {
        "model_state": model.state_dict(),
        "means": means,
        "stds": stds,
    }
    torch.save(checkpoint, path)
    print(f"Saved model checkpoint to {path}")


def load_checkpoint(model, path=MODEL_PATH):
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

def create_labels(df, horizon=10):
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = 0
    df.loc[future_return > 0.002, "label"] = 1
    df.loc[future_return < -0.002, "label"] = -1
    df = df.dropna().reset_index(drop=True)
    return df

# =====================
# DATASET
# =====================

class TradingDataset(Dataset):
    def __init__(self, df):
        self.features = df[["return", "volatility", "volume_z"]].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.features) - SEQ_LEN

    def __getitem__(self, idx):
        x = self.features[idx : idx + SEQ_LEN]
        y = self.labels[idx + SEQ_LEN]
        y_class = 0 if y == 0 else (1 if y == 1 else 2)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_class, dtype=torch.long)

# =====================
# MODEL
# =====================

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.softmax(out)

# =====================
# TRAIN
# =====================

def train_model(model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
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

# =====================
# BACKTEST
# =====================

def backtest(model, df):
    model.eval()
    features = df[["return", "volatility", "volume_z"]].values
    prices = df["close"].values
    balance = 1000
    position = 0
    entry_price = 0
    trades = []

    for i in range(SEQ_LEN, len(df)):
        x = torch.tensor(features[i - SEQ_LEN : i], dtype=torch.float32).unsqueeze(0)
        probs = model(x)[0].detach().numpy()
        long_prob = probs[1]
        short_prob = probs[2]
        price = prices[i]

        if position == 0:
            if long_prob > THRESHOLD and long_prob > short_prob:
                position = 1
                entry_price = price * (1 + FEE)
            elif short_prob > THRESHOLD and short_prob > long_prob:
                position = -1
                entry_price = price * (1 - FEE)
        else:
            if position == 1:
                change = (price - entry_price) / entry_price
                if change >= TAKE_PROFIT or change <= -STOP_LOSS:
                    exit_price = price * (1 - FEE)
                    pnl = (exit_price - entry_price) / entry_price
                    balance *= (1 + pnl)
                    trades.append(pnl)
                    position = 0
            else:
                change = (entry_price - price) / entry_price
                if change >= TAKE_PROFIT or change <= -STOP_LOSS:
                    exit_price = price * (1 + FEE)
                    pnl = (entry_price - exit_price) / entry_price
                    balance *= (1 + pnl)
                    trades.append(pnl)
                    position = 0

    print(f"Final balance: {balance:.2f}")
    print(f"Trades: {len(trades)}")
    if trades:
        print(f"Avg trade: {np.mean(trades):.4f}")
        print(f"Win rate: {np.mean([t > 0 for t in trades]):.2f}")

# =====================
# LIVE SIGNALS
# =====================

def fetch_latest_klines(symbol, interval, limit=SEQ_LEN + 1):
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
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = (df["timestamp"] // 1000).astype(int)
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_live_features(df):
    df = df.copy()
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["return"].rolling(10).std()
    df["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
        df["volume"].rolling(20).std() + 1e-9
    )
    df = df.dropna().reset_index(drop=True)
    return df


def normalize_live_features(df, means, stds):
    return normalize_features(df, means, stds)


def decode_signal(probs):
    idx = int(np.argmax(probs))
    return ["neutral", "long", "short"][idx]


def stream_live_signals(
    model,
    means,
    stds,
    symbol=REALTIME_SYMBOL,
    interval=REALTIME_INTERVAL,
    updates=REALTIME_UPDATES,
    sleep_seconds=REALTIME_SLEEP,
):
    print(f"Starting live signal stream for {symbol} on {interval} interval")
    for update in range(updates):
        raw = fetch_latest_klines(symbol, interval, limit=SEQ_LEN + 1)
        live_df = format_klines(raw)
        live_df = compute_live_features(live_df)
        live_df = normalize_live_features(live_df, means, stds)

        if len(live_df) < SEQ_LEN:
            print("Not enough live data to generate a signal yet.")
            time.sleep(sleep_seconds)
            continue

        x = torch.tensor(
            live_df[["return", "volatility", "volume_z"]].iloc[-SEQ_LEN:].values,
            dtype=torch.float32,
        ).unsqueeze(0)
        probs = model(x)[0].detach().numpy()
        signal = decode_signal(probs)
        last_price = live_df["close"].iloc[-1]
        print(f"[{update+1}/{updates}] Price: {last_price:.2f}, Signal: {signal}, probs: {probs}")
        time.sleep(sleep_seconds)

    print("Live signal stream complete.")
