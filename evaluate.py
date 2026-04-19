import pandas as pd
from trade_model import (
    LSTMModel,
    add_features,
    create_labels,
    normalize_features,
    load_checkpoint,
    predict_dataset,
    classification_report,
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

    y_true, y_pred = predict_dataset(model, test_df)
    cm, report = classification_report(y_true, y_pred)

    print("Classification report on test set:")
    for label, precision, recall, f1, support in report:
        print(
            f"{label:7}  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  support={support}"
        )

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(cm)


if __name__ == "__main__":
    main()
