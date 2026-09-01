"""Integration tests for Trade ML - testing how components work together."""

import pytest
import pandas as pd
import numpy as np
import torch
import tempfile
import os
from torch.utils.data import DataLoader

from trade_model import (
    LSTMModel,
    TradingDataset,
    train_model,
    save_checkpoint,
    load_checkpoint,
    predict_dataset,
    backtest,
    add_features,
    create_labels,
    normalize_features,
    compute_normalization_params,
    compute_class_weights,
    classification_report,
    BATCH_SIZE,
    FEATURE_COLS,
)


class TestEndToEndPipeline:
    """Integration tests for complete ML pipeline."""

    def test_pipeline_from_raw_data(self, sample_ohlcv_data):
        """Test complete pipeline from raw data to model training."""
        # Step 1: Add features
        df = add_features(sample_ohlcv_data)
        assert len(df) < len(sample_ohlcv_data)  # Some rows dropped
        assert all(col in df.columns for col in FEATURE_COLS)
        
        # Step 2: Create labels
        df = create_labels(df)
        assert "label" in df.columns
        
        # Step 3: Compute normalization params
        means, stds = compute_normalization_params(df)
        assert len(means) == len(FEATURE_COLS)
        assert len(stds) == len(FEATURE_COLS)
        
        # Step 4: Normalize features
        df = normalize_features(df, means, stds)
        
        # Step 5: Create dataset
        dataset = TradingDataset(df)
        assert len(dataset) > 0
        
        # Step 6: Create dataloader
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        batches = list(dataloader)
        assert len(batches) > 0
        
        # Step 7: Create model
        model = LSTMModel()
        assert model is not None

    def test_train_and_predict(self, trending_data):
        """Test training a model and making predictions."""
        # Prepare data
        df = add_features(trending_data)
        df = create_labels(df)
        means, stds = compute_normalization_params(df)
        df = normalize_features(df, means, stds)
        
        # Split data
        split = int(len(df) * 0.8)
        train_df = df[:split]
        test_df = df[split:]
        
        # Create dataset and dataloader
        dataset = TradingDataset(train_df)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize model
        model = LSTMModel()
        
        # Train for 1 epoch
        class_weights = compute_class_weights(train_df["label"].values)
        train_model(model, dataloader, class_weights=class_weights)
        
        # Make predictions on test set
        y_true, y_pred = predict_dataset(model, test_df)
        
        assert len(y_true) > 0
        assert len(y_pred) > 0
        assert len(y_true) == len(y_pred)
        assert all(pred in [0, 1, 2] for pred in y_pred)

    def test_checkpoint_workflow(self, trending_data, temp_dir=None):
        """Test saving and loading model checkpoints."""
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        try:
            # Prepare data
            df = add_features(trending_data)
            df = create_labels(df)
            means, stds = compute_normalization_params(df)
            df = normalize_features(df, means, stds)
            
            # Train model
            split = int(len(df) * 0.8)
            train_df = df[:split]
            dataset = TradingDataset(train_df)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            model1 = LSTMModel()
            class_weights = compute_class_weights(train_df["label"].values)
            train_model(model1, dataloader, class_weights=class_weights)
            
            # Save checkpoint
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
            save_checkpoint(model1, means, stds, checkpoint_path)
            
            # Load into new model
            model2 = LSTMModel()
            loaded_means, loaded_stds = load_checkpoint(model2, checkpoint_path)
            
            # Verify loaded parameters match
            assert loaded_means == means
            assert loaded_stds == stds
            
            # Verify both models make same predictions
            x_test = torch.randn(1, 30, len(FEATURE_COLS))
            with torch.no_grad():
                pred1 = model1(x_test)
                pred2 = model2(x_test)
            
            assert torch.allclose(pred1, pred2, atol=1e-5)
        finally:
            # Cleanup
            if temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

    def test_evaluation_metrics(self, trending_data):
        """Test that evaluation metrics are computed correctly."""
        # Prepare data
        df = add_features(trending_data)
        df = create_labels(df)
        means, stds = compute_normalization_params(df)
        df = normalize_features(df, means, stds)
        
        # Train model
        split = int(len(df) * 0.8)
        train_df = df[:split]
        test_df = df[split:]
        
        dataset = TradingDataset(train_df)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = LSTMModel()
        class_weights = compute_class_weights(train_df["label"].values)
        train_model(model, dataloader, class_weights=class_weights)
        
        # Get predictions and compute metrics
        y_true, y_pred = predict_dataset(model, test_df)
        cm, report = classification_report(y_true, y_pred)
        
        # Check metrics structure
        assert cm.shape == (3, 3)
        assert len(report) == 3
        
        # Check that metrics are in valid ranges
        for label, precision, recall, f1, support in report:
            assert 0 <= precision <= 1
            assert 0 <= recall <= 1
            assert 0 <= f1 <= 1
            assert support >= 0


class TestDataPreprocessing:
    """Integration tests for data preprocessing pipeline."""

    def test_feature_engineering_chain(self, sample_ohlcv_data):
        """Test complete feature engineering pipeline."""
        df = add_features(sample_ohlcv_data)
        
        # Verify all features are present
        for col in FEATURE_COLS:
            assert col in df.columns
            assert not df[col].isna().any()
        
        # Verify features have reasonable ranges
        assert df["return"].abs().max() < 10  # Log returns shouldn't be huge
        assert (df["rsi"] >= 0).all()
        assert (df["rsi"] <= 100).all()
        assert (df["volatility"] >= 0).all()

    def test_normalization_consistency(self, sample_ohlcv_data):
        """Test that normalization is consistent across multiple calls."""
        df = add_features(sample_ohlcv_data)
        means1, stds1 = compute_normalization_params(df)
        means2, stds2 = compute_normalization_params(df)
        
        # Should be identical
        assert means1 == means2
        assert stds1 == stds2

    def test_label_distribution(self, trending_data):
        """Test that label distribution makes sense for trending data."""
        df = add_features(trending_data)
        df = create_labels(df)
        
        # For trending data, should have some buy signals
        label_counts = df["label"].value_counts()
        
        # Should have multiple label types
        assert len(label_counts) > 1


class TestModelProperties:
    """Tests for model mathematical properties."""

    def test_model_outputs_valid_probabilities(self):
        """Test that model always outputs valid probability distributions."""
        model = LSTMModel()
        model.eval()
        
        # Test multiple batches with different data
        for _ in range(10):
            x = torch.randn(4, 30, len(FEATURE_COLS))
            with torch.no_grad():
                output = model(x)
            
            # Check shape
            assert output.shape == (4, 3)
            
            # Check probabilities sum to 1
            sums = output.sum(dim=1)
            assert torch.allclose(sums, torch.ones(4), atol=1e-5)
            
            # Check values in [0, 1]
            assert (output >= 0).all() and (output <= 1).all()

    def test_model_gradients_flow(self):
        """Test that gradients flow correctly through the model."""
        model = LSTMModel()
        x = torch.randn(2, 30, len(FEATURE_COLS))
        y = torch.tensor([0, 1], dtype=torch.long)
        
        output = model(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0), f"No gradient for {name}"


class TestDataLoaderConsistency:
    """Tests for TradingDataset consistency."""

    def test_dataset_shuffling(self, sample_labeled_data):
        """Test that dataset works with shuffled batches."""
        dataset = TradingDataset(sample_labeled_data)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Iterate through multiple epochs
        all_indices = set()
        for epoch in range(2):
            for batch_x, batch_y in dataloader:
                assert batch_x.shape[0] <= 8
                assert len(batch_y) == batch_x.shape[0]

    def test_dataset_determinism(self, sample_labeled_data):
        """Test that same data produces same samples."""
        dataset = TradingDataset(sample_labeled_data)
        
        x1, y1 = dataset[10]
        x2, y2 = dataset[10]
        
        assert torch.equal(x1, x2)
        assert y1 == y2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
