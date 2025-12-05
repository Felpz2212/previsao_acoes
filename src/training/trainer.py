"""
Training pipeline for LSTM stock prediction model.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
from loguru import logger
import json

from src.data.data_loader import StockDataLoader
from src.data.preprocessor import StockDataPreprocessor
from src.models.lstm_model import LSTMPredictor
from config.settings import (
    LSTM_SEQUENCE_LENGTH,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    get_model_path,
    get_scaler_path,
    MODELS_DIR
)


class ModelTrainer:
    """Complete training pipeline for stock prediction model."""
    
    def __init__(
        self,
        symbol: str,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        learning_rate: float = LSTM_LEARNING_RATE,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT
    ):
        """
        Initialize the trainer.
        
        Args:
            symbol: Stock ticker symbol
            sequence_length: LSTM sequence length
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.data_loader = StockDataLoader()
        self.preprocessor = None
        self.model = None
        self.metrics = {}
        
    def load_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load stock data.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with stock data
        """
        logger.info(f"Loading data for {self.symbol}")
        df = self.data_loader.load_stock_data(
            self.symbol,
            start_date,
            end_date
        )
        
        # Validate data
        self.data_loader.validate_data(df)
        
        return df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        train_split: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and split data for training.
        
        Args:
            df: DataFrame with stock data
            train_split: Fraction of data to use for training
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info("Preprocessing data")
        
        # Initialize preprocessor
        self.preprocessor = StockDataPreprocessor(
            sequence_length=self.sequence_length
        )
        
        # Fit and transform data
        X, y, df_processed = self.preprocessor.fit_transform(df)
        
        # Split into train and test sets
        split_idx = int(train_split * len(X))
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        logger.info(
            f"Train set: {len(X_train)} samples, "
            f"Test set: {len(X_test)} samples"
        )
        
        return X_train, y_train, X_test, y_test
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        logger.info("Initializing model")
        
        # Get input size from data
        input_size = X_train.shape[2]
        
        # Initialize model
        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate
        )
        
        # Train model
        logger.info("Starting training")
        self.model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=True
        )
    
    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model using multiple metrics.
        
        Args:
            X_test: Test features
            y_test: Test targets (normalized)
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Make predictions (normalized)
        predictions_scaled = self.model.predict(X_test)
        
        # Inverse transform to get real values
        predictions = np.array([
            self.preprocessor.inverse_transform_target(p) 
            for p in predictions_scaled
        ])
        y_test_real = np.array([
            self.preprocessor.inverse_transform_target(y) 
            for y in y_test
        ])
        
        # Calculate metrics with real values
        mse = np.mean((predictions - y_test_real) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test_real))
        
        # Avoid division by zero in MAPE
        mask = y_test_real != 0
        mape = np.mean(np.abs((y_test_real[mask] - predictions[mask]) / y_test_real[mask])) * 100
        
        # R-squared
        ss_res = np.sum((y_test_real - predictions) ** 2)
        ss_tot = np.sum((y_test_real - np.mean(y_test_real)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Directional accuracy (did we predict the direction correctly?)
        direction_actual = np.diff(y_test_real) > 0
        direction_pred = np.diff(predictions) > 0
        directional_accuracy = np.mean(direction_actual == direction_pred) * 100
        
        self.metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy),
            'test_samples': len(y_test)
        }
        
        logger.info("Evaluation Metrics:")
        logger.info(f"  RMSE: ${rmse:.2f}")
        logger.info(f"  MAE: ${mae:.2f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        return self.metrics
    
    def save_model(self):
        """Save the trained model and preprocessor."""
        logger.info("Saving model and preprocessor")
        
        # Save model
        model_path = get_model_path(self.symbol)
        self.model.save(model_path)
        
        # Save preprocessor
        scaler_path = get_scaler_path(self.symbol)
        self.preprocessor.save(scaler_path)
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'trained_at': datetime.now().isoformat(),
            'sequence_length': self.sequence_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'metrics': self.metrics,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path)
        }
        
        metadata_path = MODELS_DIR / f"metadata_{self.symbol}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def run_training_pipeline(
        self,
        start_date: str,
        end_date: str,
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> Dict[str, float]:
        """
        Run the complete training pipeline.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            train_split: Fraction of data for training
            val_split: Fraction of training data for validation
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Starting training pipeline for {self.symbol}")
        
        # Load data
        df = self.load_data(start_date, end_date)
        
        # Prepare data
        X_train_full, y_train_full, X_test, y_test = self.prepare_data(
            df, train_split
        )
        
        # Split training data into train and validation
        if val_split > 0:
            val_samples = int(len(X_train_full) * val_split)
            X_train = X_train_full[:-val_samples]
            y_train = y_train_full[:-val_samples]
            X_val = X_train_full[-val_samples:]
            y_val = y_train_full[-val_samples:]
        else:
            X_train = X_train_full
            y_train = y_train_full
            X_val = None
            y_val = None
        
        # Train model
        self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # Save model
        self.save_model()
        
        logger.info("Training pipeline completed successfully")
        
        return metrics


def train_stock_model(
    symbol: str,
    start_date: str,
    end_date: str,
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function to train a model for a stock.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        **kwargs: Additional arguments for ModelTrainer
        
    Returns:
        Dictionary with evaluation metrics
    """
    trainer = ModelTrainer(symbol, **kwargs)
    metrics = trainer.run_training_pipeline(start_date, end_date)
    return metrics


if __name__ == "__main__":
    # Example: Train model for Apple stock
    from config.settings import DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_STOCK_SYMBOL
    
    logger.info(f"Training model for {DEFAULT_STOCK_SYMBOL}")
    
    metrics = train_stock_model(
        symbol=DEFAULT_STOCK_SYMBOL,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE
    )
    
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

