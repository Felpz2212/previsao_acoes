"""
Tests for LSTM model functionality.
"""
import pytest
import numpy as np
import torch
from src.models.lstm_model import LSTMModel, LSTMPredictor

# Tentar importar ImprovedLSTMPredictor (versão completa com fit())
try:
    from src.training.improved_trainer import ImprovedLSTMPredictor
    HAS_IMPROVED = True
except ImportError:
    # Fallback para versão simplificada (sem fit)
    try:
        from railway_app.backend.core.improved_lstm import ImprovedLSTMPredictor
        HAS_IMPROVED = True
    except ImportError:
        HAS_IMPROVED = False


def test_lstm_model_initialization():
    """Test LSTM model initialization."""
    model = LSTMModel(input_size=10, hidden_size=50, num_layers=2)
    assert model is not None
    assert model.input_size == 10
    assert model.hidden_size == 50
    assert model.num_layers == 2


def test_lstm_model_forward():
    """Test forward pass through LSTM."""
    model = LSTMModel(input_size=10, hidden_size=50, num_layers=2)
    
    # Create dummy input
    batch_size = 32
    sequence_length = 60
    x = torch.randn(batch_size, sequence_length, 10)
    
    # Forward pass
    output, hidden = model(x)
    
    assert output.shape == (batch_size, 1)
    assert len(hidden) == 2  # (h_n, c_n)


def test_lstm_predictor_initialization():
    """Test LSTM predictor initialization."""
    predictor = LSTMPredictor(input_size=10)
    assert predictor is not None
    assert predictor.model is not None
    assert predictor.device in ['cpu', 'cuda']


def test_lstm_predictor_training():
    """Test model training."""
    # Create dummy data
    X_train = np.random.randn(100, 60, 10)
    y_train = np.random.randn(100)
    
    predictor = LSTMPredictor(input_size=10)
    
    # Train for a few epochs
    predictor.fit(X_train, y_train, epochs=2, batch_size=32, verbose=False)
    
    assert len(predictor.train_losses) == 2


def test_lstm_predictor_prediction():
    """Test making predictions."""
    # Create dummy data
    X_test = np.random.randn(10, 60, 10)
    
    predictor = LSTMPredictor(input_size=10)
    predictions = predictor.predict(X_test)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 10


# ==================== IMPROVED LSTM TESTS ====================

@pytest.mark.skipif(not HAS_IMPROVED, reason="ImprovedLSTMPredictor not available")
def test_improved_lstm_initialization():
    """Test ImprovedLSTMPredictor initialization."""
    predictor = ImprovedLSTMPredictor(
        input_size=16,
        hidden_size=64,
        num_layers=3,
        dropout=0.3
    )
    assert predictor is not None
    assert predictor.model is not None


@pytest.mark.skipif(not HAS_IMPROVED, reason="ImprovedLSTMPredictor not available")
def test_improved_lstm_training():
    """Test ImprovedLSTMPredictor training with early stopping."""
    X_train = np.random.randn(100, 60, 16).astype(np.float32)
    y_train = np.random.randn(100).astype(np.float32)
    X_val = np.random.randn(20, 60, 16).astype(np.float32)
    y_val = np.random.randn(20).astype(np.float32)
    
    predictor = ImprovedLSTMPredictor(input_size=16)
    
    # Train for a few epochs
    predictor.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=3,
        patience=5
    )
    
    assert len(predictor.train_losses) > 0


@pytest.mark.skipif(not HAS_IMPROVED, reason="ImprovedLSTMPredictor not available")
def test_improved_lstm_prediction():
    """Test ImprovedLSTMPredictor predictions."""
    X_test = np.random.randn(10, 60, 16).astype(np.float32)
    
    predictor = ImprovedLSTMPredictor(input_size=16)
    predictions = predictor.predict(X_test)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 10

