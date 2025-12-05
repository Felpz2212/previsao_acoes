"""
Tests for data loading functionality.
"""
import pytest
import pandas as pd
from src.data.data_loader import StockDataLoader


def test_stock_data_loader_initialization():
    """Test StockDataLoader initialization."""
    loader = StockDataLoader()
    assert loader is not None


def test_load_stock_data():
    """Test loading stock data from Yahoo Finance."""
    loader = StockDataLoader()
    df = loader.load_stock_data("AAPL", "2023-01-01", "2023-12-31")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'close' in df.columns
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'volume' in df.columns
    assert 'timestamp' in df.columns


def test_validate_data():
    """Test data validation."""
    loader = StockDataLoader()
    df = loader.load_stock_data("AAPL", "2023-01-01", "2023-12-31")
    
    # Should not raise any exception
    assert loader.validate_data(df) == True


def test_get_latest_price():
    """Test getting latest price."""
    loader = StockDataLoader()
    latest = loader.get_latest_price("AAPL")
    
    assert isinstance(latest, dict)
    assert 'symbol' in latest
    assert latest['symbol'] == 'AAPL'

