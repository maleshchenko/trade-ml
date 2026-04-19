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
    df = pd.read_csv("data.csv")
    df = add_features(df)
    df = create_labels(df)
    means, stds = compute_normalization_params(df)
    df = normalize_features(df, means, stds)

    split = int(len(df) * 0.8)
    train_df = df[:split]

    model = LSTMModel()

    if os.path.exists(MODEL_PATH) and not retrain:
        print(f"Model checkpoint already exists at {MODEL_PATH}.")
        print("Use --retrain to force training or remove the checkpoint file.")
        return

    train_dataset = TradingDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    class_weights = compute_class_weights(train_df["label"].values)

    train_model(model, train_loader, class_weights=class_weights)
    save_checkpoint(model, means, stds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the trading model.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the model even if a saved checkpoint exists.",
    )
    args = parser.parse_args()
    main(retrain=args.retrain)
