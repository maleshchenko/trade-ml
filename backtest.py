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
    df = pd.read_csv("data.csv")
    df = add_features(df)
    df = create_labels(df)

    model = LSTMModel()
    means, stds = load_checkpoint(model)

    df = normalize_features(df, means, stds)
    split = int(len(df) * 0.8)
    test_df = df[split:]

    print("Backtesting loaded model on test data...")
    backtest(model, test_df)


if __name__ == "__main__":
    main()
