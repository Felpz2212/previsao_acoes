"""
Tests for data preprocessing functionality.
"""
import pytest
import numpy as np
import pandas as pd
from src.data.preprocessor import StockDataPreprocessor
from src.data.data_loader import StockDataLoader


@pytest.fixture
def sample_data():
    """Load sample stock data for testing."""
    loader = StockDataLoader()
    df = loader.load_stock_data("AAPL", "2023-01-01", "2023-12-31")
    return df


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = StockDataPreprocessor(sequence_length=60)
    assert preprocessor.sequence_length == 60
    assert preprocessor.scaler is not None


def test_prepare_features(sample_data):
    """Test feature engineering."""
    preprocessor = StockDataPreprocessor()
    df_features = preprocessor.prepare_features(sample_data)
    
    assert isinstance(df_features, pd.DataFrame)
    assert 'ma_7' in df_features.columns
    assert 'ma_30' in df_features.columns
    assert 'volatility_7' in df_features.columns
    assert len(df_features) == len(sample_data)


def test_fit_transform(sample_data):
    """Test fit and transform."""
    preprocessor = StockDataPreprocessor(sequence_length=60)
    X, y, df_processed = preprocessor.fit_transform(sample_data)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X.shape) == 3  # (samples, sequence_length, features)
    assert X.shape[1] == 60  # sequence_length
    assert len(X) == len(y)


def test_create_sequences():
    """Test sequence creation."""
    preprocessor = StockDataPreprocessor(sequence_length=10)
    
    # Create dummy data
    data = np.random.randn(100, 5)
    target = np.random.randn(100)
    
    X, y = preprocessor.create_sequences(data, target)
    
    assert len(X) == 90  # 100 - sequence_length
    assert X.shape[1] == 10  # sequence_length
    assert X.shape[2] == 5  # features
    assert len(y) == 90

