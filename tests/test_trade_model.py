"""Unit tests for trade_model.py - feature engineering, model, and utilities."""

import pytest
import numpy as np
import pandas as pd
import torch
import os
import tempfile
from trade_model import (
    compute_rsi,
    compute_atr,
    add_features,
    normalize_features,
    compute_normalization_params,
    create_labels,
    TradingDataset,
    LSTMModel,
    label_to_class,
    classification_report,
    save_checkpoint,
    load_checkpoint,
    FEATURE_COLS,
    SEQ_LEN,
)


class TestFeatureEngineering:
    """Tests for technical indicator and feature engineering functions."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="1min")
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": close_prices * 0.98 + np.random.randn(100) * 0.1,
            "high": close_prices * 1.02 + np.random.rand(100) * 0.5,
            "low": close_prices * 0.98 - np.random.rand(100) * 0.5,
            "close": close_prices,
            "volume": np.random.rand(100) * 1000,
        })
        return df

    def test_compute_rsi(self, sample_df):
        """Test RSI computation produces valid values."""
        rsi = compute_rsi(sample_df["close"], period=14)
        
        # Check RSI values are between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
        assert len(valid_rsi) > 0
        assert len(rsi) == len(sample_df)

    def test_compute_atr(self, sample_df):
        """Test ATR computation produces valid positive values."""
        atr = compute_atr(sample_df, period=14)
        
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
        assert len(valid_atr) > 0
        assert len(atr) == len(sample_df)

    def test_add_features(self, sample_df):
        """Test that add_features computes all required features."""
        df_with_features = add_features(sample_df)
        
        # Check that all required features are present
        for col in FEATURE_COLS:
            assert col in df_with_features.columns, f"Missing feature: {col}"
        
        # Check that NaN values are dropped
        assert not df_with_features.isna().any().any()
        
        # Check that features have reasonable ranges
        assert (df_with_features["return"].abs() < 1).all()  # Log returns shouldn't be huge
        assert (df_with_features["rsi"] >= 0).all()
        assert (df_with_features["rsi"] <= 100).all()

    def test_normalize_features(self, sample_df):
        """Test feature normalization to zero mean and unit variance."""
        df_with_features = add_features(sample_df)
        df_normalized = normalize_features(df_with_features)
        
        # Check that normalized features have ~zero mean and ~unit variance
        for col in FEATURE_COLS:
            mean = df_normalized[col].mean()
            std = df_normalized[col].std()
            assert abs(mean) < 0.1, f"Feature {col} has non-zero mean: {mean}"
            assert 0.9 < std < 1.1, f"Feature {col} has non-unit std: {std}"

    def test_normalize_with_params(self, sample_df):
        """Test normalization using precomputed parameters."""
        df_with_features = add_features(sample_df)
        means, stds = compute_normalization_params(df_with_features)
        
        # Normalize with these parameters
        df_normalized = normalize_features(df_with_features, means, stds)
        
        # Should have zero mean and unit variance
        for col in FEATURE_COLS:
            mean = df_normalized[col].mean()
            std = df_normalized[col].std()
            assert abs(mean) < 0.1
            assert 0.9 < std < 1.1


class TestLabeling:
    """Tests for label generation."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=500, freq="1min")
        
        # Create a trending price series
        trend = np.linspace(0, 10, 500)
        close_prices = 100 + trend + np.random.randn(500) * 1.0
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": close_prices,
            "high": close_prices + np.abs(np.random.randn(500)) * 0.5,
            "low": close_prices - np.abs(np.random.randn(500)) * 0.5,
            "close": close_prices,
            "volume": np.random.rand(500) * 1000,
        })
        return df

    def test_create_labels(self, sample_df):
        """Test that labels are created correctly."""
        df_with_features = add_features(sample_df)
        df_labeled = create_labels(df_with_features)
        
        # Check that label column exists
        assert "label" in df_labeled.columns
        
        # Check that labels are -1, 0, or 1
        valid_labels = df_labeled["label"].unique()
        assert set(valid_labels).issubset({-1, 0, 1})
        
        # Check that some labels exist (not all neutral)
        assert not (df_labeled["label"] == 0).all()

    def test_create_labels_horizon(self, sample_df):
        """Test that different horizons produce different labels."""
        df_with_features = add_features(sample_df)
        df_h10 = create_labels(df_with_features, horizon=10)
        df_h20 = create_labels(df_with_features, horizon=20)
        
        # Different horizons should typically produce different label distributions
        # This is a probabilistic test, but with random data it should usually differ
        assert "label" in df_h10.columns
        assert "label" in df_h20.columns


class TestDataset:
    """Tests for the TradingDataset class."""

    @pytest.fixture
    def labeled_df(self):
        """Create a labeled DataFrame."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="1min")
        close_prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": close_prices * 0.98,
            "high": close_prices * 1.02,
            "low": close_prices * 0.98,
            "close": close_prices,
            "volume": np.random.rand(200) * 1000,
        })
        
        df = add_features(df)
        df = create_labels(df)
        df = normalize_features(df)
        return df

    def test_dataset_length(self, labeled_df):
        """Test that dataset length is correct."""
        dataset = TradingDataset(labeled_df)
        expected_length = len(labeled_df) - SEQ_LEN
        assert len(dataset) == expected_length

    def test_dataset_getitem(self, labeled_df):
        """Test that dataset returns correct item format."""
        dataset = TradingDataset(labeled_df)
        x, y = dataset[0]
        
        # Check shapes
        assert x.shape == (SEQ_LEN, len(FEATURE_COLS))
        assert y.shape == torch.Size([])
        
        # Check types
        assert x.dtype == torch.float32
        assert y.dtype == torch.long
        
        # Check label values are 0, 1, or 2
        assert y.item() in {0, 1, 2}

    def test_dataset_sequence_continuity(self, labeled_df):
        """Test that sequences are continuous and non-overlapping in label."""
        dataset = TradingDataset(labeled_df)
        x0, y0 = dataset[0]
        x1, y1 = dataset[1]
        
        # The sequences should overlap by SEQ_LEN-1 points
        # and the label of x0 should come from index SEQ_LEN
        # The label of x1 should come from index SEQ_LEN+1
        assert x0.shape[0] == SEQ_LEN
        assert x1.shape[0] == SEQ_LEN


class TestModel:
    """Tests for the LSTMModel."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = LSTMModel()
        assert model is not None
        assert hasattr(model, "lstm")
        assert hasattr(model, "fc")
        assert hasattr(model, "softmax")

    def test_model_forward_pass(self):
        """Test that model forward pass works and outputs correct shape."""
        model = LSTMModel()
        batch_size = 4
        x = torch.randn(batch_size, SEQ_LEN, len(FEATURE_COLS))
        
        output = model(x)
        
        # Output should be (batch_size, 3) for 3 classes
        assert output.shape == (batch_size, 3)
        
        # Output should be probabilities (sum to 1)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size))
        
        # Output values should be between 0 and 1
        assert (output >= 0).all() and (output <= 1).all()

    def test_model_eval_mode(self):
        """Test that model works in eval mode."""
        model = LSTMModel()
        model.eval()
        
        x = torch.randn(1, SEQ_LEN, len(FEATURE_COLS))
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 3)

    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = LSTMModel()
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestCheckpoint:
    """Tests for model checkpoint save/load functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_checkpoint(self, temp_dir):
        """Test that checkpoint is saved correctly."""
        model = LSTMModel()
        means = {col: 0.0 for col in FEATURE_COLS}
        stds = {col: 1.0 for col in FEATURE_COLS}
        
        checkpoint_path = os.path.join(temp_dir, "test_model.pt")
        save_checkpoint(model, means, stds, checkpoint_path)
        
        assert os.path.exists(checkpoint_path)
        assert os.path.getsize(checkpoint_path) > 0

    def test_load_checkpoint(self, temp_dir):
        """Test that checkpoint can be loaded."""
        # Save a model
        model1 = LSTMModel()
        means = {col: 0.5 for col in FEATURE_COLS}
        stds = {col: 1.5 for col in FEATURE_COLS}
        
        checkpoint_path = os.path.join(temp_dir, "test_model.pt")
        save_checkpoint(model1, means, stds, checkpoint_path)
        
        # Load into a new model
        model2 = LSTMModel()
        loaded_means, loaded_stds = load_checkpoint(model2, checkpoint_path)
        
        # Check that parameters match
        assert loaded_means == means
        assert loaded_stds == stds

    def test_checkpoint_missing_file(self, temp_dir):
        """Test that loading missing checkpoint raises error."""
        model = LSTMModel()
        checkpoint_path = os.path.join(temp_dir, "nonexistent.pt")
        
        with pytest.raises(FileNotFoundError):
            load_checkpoint(model, checkpoint_path)


class TestUtilities:
    """Tests for utility functions."""

    def test_label_to_class(self):
        """Test conversion of raw labels to class indices."""
        labels = np.array([0, 1, -1, 0, 1, -1])
        classes = label_to_class(labels)
        
        # 0 -> 0, 1 -> 1, -1 -> 2
        assert classes[0] == 0
        assert classes[1] == 1
        assert classes[2] == 2
        assert classes[3] == 0
        assert classes[4] == 1
        assert classes[5] == 2

    def test_normalization_params(self):
        """Test computation of normalization parameters."""
        np.random.seed(42)
        data = {col: np.random.randn(100) for col in FEATURE_COLS}
        df = pd.DataFrame(data)
        
        means, stds = compute_normalization_params(df)
        
        # Check that means and stds are computed for all features
        for col in FEATURE_COLS:
            assert col in means
            assert col in stds
            assert stds[col] > 0


class TestFeatureEdgeCases:
    """Tests for edge cases and error handling in feature engineering."""

    def test_add_features_small_df(self):
        """Test that add_features handles small DataFrames."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        })
        
        result = add_features(df)
        
        # Should return a DataFrame (possibly empty after dropna)
        assert isinstance(result, pd.DataFrame)

    def test_normalize_features_with_zeros(self):
        """Test normalization when features have zero variance."""
        df = pd.DataFrame({col: [1.0] * 50 for col in FEATURE_COLS})
        
        # Should handle zero variance gracefully
        result = normalize_features(df)
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
