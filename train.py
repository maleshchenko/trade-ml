"""Train the LSTM trading model on historical data.

This script:
1. Loads historical market data
2. Computes technical features
3. Creates training labels
4. Trains the LSTM model with class weighting for imbalanced data
5. Saves the trained model and normalization parameters

Usage:
    python train.py              # Train (skip if checkpoint exists)
    python train.py --retrain    # Force retrain even if checkpoint exists
"""

import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader
from trade_model import (
    BATCH_SIZE,
    TradingDataset,
    LSTMModel,
    add_features,
    create_labels,
    compute_normalization_params,
    normalize_features,
    train_model,
    save_checkpoint,
    compute_class_weights,
    MODEL_PATH,
)


def main(retrain: bool = False):
    """Train the trading model.
    
    Args:
        retrain: If True, retrain even if checkpoint exists.
    """
    # Load raw historical data
    df = pd.read_csv("data.csv")
    
    # Add technical indicators and features
    df = add_features(df)
    
    # Create training labels based on future price movements
    df = create_labels(df)
    
    # Compute normalization parameters (mean and std) from data
    means, stds = compute_normalization_params(df)
    
    # Normalize features to zero mean and unit variance
    df = normalize_features(df, means, stds)

    # Split data: 80% train, 20% test
    split = int(len(df) * 0.8)
    train_df = df[:split]

    # Initialize model
    model = LSTMModel()

    # Check if model already exists
    if os.path.exists(MODEL_PATH) and not retrain:
        print(f"Model checkpoint already exists at {MODEL_PATH}.")
        print("Use --retrain to force training or remove the checkpoint file.")
        return

    # Create dataset and data loader for training
    train_dataset = TradingDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Compute class weights to handle imbalanced labels
    class_weights = compute_class_weights(train_df["label"].values)

    # Train the model
    train_model(model, train_loader, class_weights=class_weights)
    
    # Save trained model and normalization parameters
    save_checkpoint(model, means, stds)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the trading model.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the model even if a saved checkpoint exists.",
    )
    args = parser.parse_args()
    main(retrain=args.retrain)
