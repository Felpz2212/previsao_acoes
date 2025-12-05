"""
Data preprocessing and feature engineering for LSTM model.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import joblib
from loguru import logger
from pathlib import Path


class StockDataPreprocessor:
    """Preprocess stock data for LSTM model training and prediction."""
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize the preprocessor.
        
        Args:
            sequence_length: Number of time steps to use for LSTM input
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))  # Separate scaler for target
        self.feature_columns = None
        self.target_column = 'close'
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for the model.
        
        Args:
            df: Raw stock data DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Sort by timestamp to ensure correct order
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['low']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        df['ma_7'] = df['close'].rolling(window=7, min_periods=1).mean()
        df['ma_30'] = df['close'].rolling(window=30, min_periods=1).mean()
        df['ma_90'] = df['close'].rolling(window=90, min_periods=1).mean()
        
        # Volatility
        df['volatility_7'] = df['close'].rolling(window=7, min_periods=1).std()
        df['volatility_30'] = df['close'].rolling(window=30, min_periods=1).std()
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_7'] = df['volume'].rolling(window=7, min_periods=1).mean()
        
        # Momentum indicators
        df['momentum'] = df['close'] - df['close'].shift(4)
        
        # Fill NaN values from rolling calculations
        df = df.bfill().ffill()
        
        return df
    
    def create_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Feature data array
            target: Target values array
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Fit the scaler and transform the data.
        
        Args:
            df: DataFrame with stock data
            feature_cols: List of feature columns to use (if None, uses default set)
            
        Returns:
            Tuple of (X_sequences, y_targets, processed_df)
        """
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Define feature columns if not provided
        if feature_cols is None:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'price_change', 'high_low_pct', 'close_open_pct',
                'ma_7', 'ma_30', 'ma_90',
                'volatility_7', 'volatility_30',
                'volume_change', 'volume_ma_7', 'momentum'
            ]
        else:
            self.feature_columns = feature_cols
        
        # Ensure all feature columns exist
        missing_cols = set(self.feature_columns) - set(df_processed.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        # Extract features and target
        features = df_processed[self.feature_columns].values
        target = df_processed[self.target_column].values.reshape(-1, 1)
        
        # Fit and transform features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit and transform target
        target_scaled = self.target_scaler.fit_transform(target).flatten()
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, target_scaled)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y, df_processed
    
    def transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Transform data using fitted scaler.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Tuple of (X_sequences, y_targets, processed_df)
        """
        if self.feature_columns is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Extract features and target
        features = df_processed[self.feature_columns].values
        target = df_processed[self.target_column].values.reshape(-1, 1)
        
        # Transform features
        features_scaled = self.scaler.transform(features)
        
        # Transform target
        target_scaled = self.target_scaler.transform(target).flatten()
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, target_scaled)
        
        return X, y, df_processed
    
    def transform_for_prediction(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """
        Transform the most recent data for prediction.
        
        Args:
            df: DataFrame with stock data (must have at least sequence_length rows)
            
        Returns:
            Scaled sequence array ready for prediction
        """
        if len(df) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} rows, got {len(df)}"
            )
        
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Extract features
        features = df_processed[self.feature_columns].values
        
        # Transform features
        features_scaled = self.scaler.transform(features)
        
        # Get the last sequence
        X = features_scaled[-self.sequence_length:]
        
        # Reshape for model input (batch_size=1, sequence_length, n_features)
        X = X.reshape(1, self.sequence_length, -1)
        
        return X
    
    def inverse_transform_target(self, scaled_value: float) -> float:
        """
        Inverse transform a scaled target value back to original scale.
        
        Args:
            scaled_value: Scaled prediction value
            
        Returns:
            Value in original scale
        """
        # Use the dedicated target scaler
        scaled_array = np.array([[scaled_value]])
        inverse = self.target_scaler.inverse_transform(scaled_array)
        return inverse[0, 0]
    
    def save(self, path: Path):
        """
        Save the preprocessor state.
        
        Args:
            path: Path to save the preprocessor
        """
        state = {
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'target_column': self.target_column
        }
        joblib.dump(state, path)
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'StockDataPreprocessor':
        """
        Load a saved preprocessor.
        
        Args:
            path: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor instance
        """
        state = joblib.load(path)
        
        preprocessor = cls(sequence_length=state['sequence_length'])
        preprocessor.scaler = state['scaler']
        preprocessor.target_scaler = state.get('target_scaler', MinMaxScaler())  # Backward compatibility
        preprocessor.feature_columns = state['feature_columns']
        preprocessor.target_column = state['target_column']
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


# Removido bloco de teste para deploy

