"""Evaluate the trained trading model on the test set.

This script:
1. Loads the trained model and normalization parameters
2. Generates predictions on the test set (20% of data)
3. Computes classification metrics (precision, recall, F1-score)
4. Displays confusion matrix and per-class performance

Usage:
    python evaluate.py
"""

import logging
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
    """Evaluate model performance on test set."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load and preprocess data
    df = pd.read_csv("data.csv")
    df = add_features(df)
    df = create_labels(df)

    # Initialize model and load trained weights
    model = LSTMModel()
    means, stds = load_checkpoint(model)

    # Normalize features using training data statistics
    df = normalize_features(df, means, stds)
    
    # Split data: 80% train, 20% test
    split = int(len(df) * 0.8)
    test_df = df[split:]

    # Generate predictions and get true labels
    y_true, y_pred = predict_dataset(model, test_df)
    
    # Compute classification metrics
    cm, report = classification_report(y_true, y_pred)

    # Log classification report
    logger.info("Classification report on test set:")
    for label, precision, recall, f1, support in report:
        logger.info(
            f"{label:7}  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  support={support}"
        )

    # Log confusion matrix
    logger.info("\nConfusion matrix (rows=true, cols=predicted):")
    logger.info(cm)


if __name__ == "__main__":
    main()
