import pandas as pd
import numpy as np
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

# =====================
# FEATURE ENGINEERING
# =====================
# Function to add technical features to the dataframe
def add_features(df):
    df = df.copy()

    # Calculate logarithmic returns
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    # Calculate rolling volatility (standard deviation of returns over 10 periods)
    df["volatility"] = df["return"].rolling(10).std()
    # Calculate z-score of volume over 20 periods
    df["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)

    # Drop rows with NaN values and reset index
    df = df.dropna().reset_index(drop=True)
    return df

# Function to normalize features using z-score normalization
def normalize_features(df):
    cols = ["return", "volatility", "volume_z"]

    for col in cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-9
        df[col] = (df[col] - mean) / std

    return df

# =====================
# LABELING
# =====================
# def create_labels(df, horizon=10):
#     df = df.copy()
#     future_max = df["high"].rolling(horizon).max().shift(-horizon)
#     future_min = df["low"].rolling(horizon).min().shift(-horizon)

#     entry_price = df["close"]

#     up_move = (future_max - entry_price) / entry_price
#     down_move = (entry_price - future_min) / entry_price

#     df["label"] = (up_move > TAKE_PROFIT) & (down_move < STOP_LOSS)
#     df["label"] = df["label"].astype(int)

#     df = df.dropna().reset_index(drop=True)
#     return df

# =====================
# LABELING
# =====================
# Function to create labels for the dataset based on future price movements
def create_labels(df, horizon=10):
    df = df.copy()

    # Calculate future return over the horizon period
    future_return = df["close"].shift(-horizon) / df["close"] - 1

    # Initialize labels: 0 for neutral, 1 for long, -1 for short
    df["label"] = 0
    # Label as long if future return > 0.2%
    df.loc[future_return > 0.002, "label"] = 1
    # Label as short if future return < -0.2%
    df.loc[future_return < -0.002, "label"] = -1

    # Drop rows with NaN and reset index
    df = df.dropna().reset_index(drop=True)
    return df

# =====================
# DATASET
# =====================
# Custom dataset class for trading data
class TradingDataset(Dataset):
    def __init__(self, df):
        # Extract features and labels from dataframe
        self.features = df[["return", "volatility", "volume_z"]].values
        self.labels = df["label"].values

    def __len__(self):
        # Return length minus sequence length to account for sliding window
        return len(self.features) - SEQ_LEN

    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx+SEQ_LEN]
        # Get label at the end of the sequence
        y = self.labels[idx+SEQ_LEN]
        # Convert labels to class indices: 0=neutral, 1=long, 2=short
        y_class = 0 if y == 0 else (1 if y == 1 else 2)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_class, dtype=torch.long)

# =====================
# MODEL
# =====================
# LSTM model for predicting trading signals
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        # LSTM layer with input size 3 (features), hidden size 64
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Fully connected layer to output 3 classes
        self.fc = nn.Linear(hidden_size, 3)  # 3 classes: neutral, long, short
        # Softmax for probability distribution
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass through LSTM
        out, _ = self.lstm(x)
        # Take the last output of the sequence
        out = out[:, -1, :]
        # Pass through fully connected layer
        out = self.fc(out)
        # Apply softmax
        return self.softmax(out)

# =====================
# TRAIN
# =====================
# Function to train the model
def train_model(model, dataloader):
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Set model to training mode
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0

        for x, y in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            preds = model(x)
            # Calculate loss
            loss = criterion(preds, y)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()

            total_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =====================
# BACKTEST
# =====================
# Function to backtest the trained model on test data
def backtest(model, df):
    # Set model to evaluation mode
    model.eval()

    # Extract features and prices
    features = df[["return", "volatility", "volume_z"]].values
    prices = df["close"].values

    # Initialize balance and position
    balance = 1000
    position = 0  # 0=none, 1=long, -1=short
    entry_price = 0

    # List to store trade PnLs
    trades = []

    # Loop through each time step starting from SEQ_LEN
    for i in range(SEQ_LEN, len(df)):
        # Prepare input sequence
        x = torch.tensor(features[i-SEQ_LEN:i], dtype=torch.float32).unsqueeze(0)
        # Get model predictions (probabilities)
        probs = model(x)[0].detach().numpy()  # [neutral, long, short]
        long_prob = probs[1]
        short_prob = probs[2]

        price = prices[i]

        # Print probabilities every 1000 steps
        if i % 1000 == 0:
            print(f"Long: {long_prob:.4f}, Short: {short_prob:.4f}")

        if position == 0:
            # Enter long if long_prob > threshold and higher than short_prob
            if long_prob > THRESHOLD and long_prob > short_prob:
                position = 1
                entry_price = price * (1 + FEE)  # Account for entry fee
            # Enter short if short_prob > threshold and higher than long_prob
            elif short_prob > THRESHOLD and short_prob > long_prob:
                position = -1
                entry_price = price * (1 - FEE)  # Account for entry fee
        else:
            if position == 1:  # Long position
                # Calculate price change
                change = (price - entry_price) / entry_price
                # Exit if take profit or stop loss hit
                if change >= TAKE_PROFIT or change <= -STOP_LOSS:
                    exit_price = price * (1 - FEE)  # Account for exit fee
                    pnl = (exit_price - entry_price) / entry_price  # Profit/loss
                    balance *= (1 + pnl)  # Update balance
                    trades.append(pnl)  # Record trade
                    position = 0  # Reset position
            else:  # Short position (position == -1)
                # Calculate price change for short
                change = (entry_price - price) / entry_price
                # Exit if take profit or stop loss hit
                if change >= TAKE_PROFIT or change <= -STOP_LOSS:
                    exit_price = price * (1 + FEE)  # Account for exit fee
                    pnl = (entry_price - exit_price) / entry_price  # Profit/loss
                    balance *= (1 + pnl)  # Update balance
                    trades.append(pnl)  # Record trade
                    position = 0  # Reset position

    # Print backtest results
    print(f"Final balance: {balance:.2f}")
    print(f"Trades: {len(trades)}")

    if trades:
        print(f"Avg trade: {np.mean(trades):.4f}")
        print(f"Win rate: {np.mean([t > 0 for t in trades]):.2f}")

# =====================
# MAIN
# =====================
# Main function to run the training and backtesting pipeline
def main():
    # Load data from CSV
    df = pd.read_csv("data.csv")

    # Add features to the dataframe
    df = add_features(df)
    # Create labels
    df = create_labels(df)
    # Normalize features
    df = normalize_features(df)

    # Split data into train and test (80/20)
    split = int(len(df) * 0.8)
    train_df = df[:split]
    test_df = df[split:]

    # Create dataset and dataloader for training
    train_dataset = TradingDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the model
    model = LSTMModel()

    # Train the model
    train_model(model, train_loader)

    # Backtest on test data
    print("\nBacktesting...\n")
    backtest(model, test_df)


if __name__ == "__main__":
    main()
