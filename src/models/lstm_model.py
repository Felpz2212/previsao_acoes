"""
LSTM Model for stock price prediction using PyTorch.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from loguru import logger


class LSTMModel(nn.Module):
    """LSTM Neural Network for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            output_size: Number of output values (1 for price prediction)
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            Tuple of (output, (h_n, c_n))
        """
        # LSTM layer
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        output = self.fc(last_output)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on ('cpu' or 'cuda')
            
        Returns:
            Tuple of (h_0, c_0) tensors
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)


class LSTMPredictor:
    """Wrapper class for training and using LSTM model."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize the LSTM predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            device: Device to use ('cpu' or 'cuda')
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 32
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            X_train: Training features
            y_train: Training targets
            batch_size: Batch size for training
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Create batches
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Get batch data
            X_batch = torch.FloatTensor(X_train[batch_indices]).to(self.device)
            y_batch = torch.FloatTensor(y_train[batch_indices]).to(self.device)
            y_batch = y_batch.reshape(-1, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32
    ) -> float:
        """
        Validate the model.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size for validation
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                # Get batch data
                X_batch = torch.FloatTensor(X_val[i:i + batch_size]).to(self.device)
                y_batch = torch.FloatTensor(y_val[i:i + batch_size]).to(self.device)
                y_batch = y_batch.reshape(-1, 1)
                
                # Forward pass
                outputs, _ = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of epochs to train
            batch_size: Batch size
            verbose: Whether to print progress
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            self.train_losses.append(train_loss)
            
            # Validate
            if X_val is not None and y_val is not None:
                val_loss = self.validate(X_val, y_val, batch_size)
                self.val_losses.append(val_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}"
                    )
        
        logger.info("Training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions, _ = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        return predictions.flatten()
    
    def save(self, path: Path):
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> 'LSTMPredictor':
        """
        Load a saved model.
        
        Args:
            path: Path to load the model from
            device: Device to load the model on
            
        Returns:
            Loaded LSTMPredictor instance
        """
        checkpoint = torch.load(path, map_location=device or 'cpu')
        
        predictor = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            learning_rate=checkpoint['learning_rate'],
            device=device
        )
        
        predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        predictor.train_losses = checkpoint.get('train_losses', [])
        predictor.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Model loaded from {path}")
        return predictor


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    sequence_length = 60
    input_size = 16
    
    # Create dummy data
    X = np.random.randn(1000, sequence_length, input_size)
    y = np.random.randn(1000)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Initialize predictor
    predictor = LSTMPredictor(input_size=input_size)
    
    # Train
    predictor.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=32
    )
    
    # Predict
    predictions = predictor.predict(X_val[:5])
    print(f"Predictions: {predictions}")
    print(f"Actual: {y_val[:5]}")

