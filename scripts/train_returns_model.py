#!/usr/bin/env python3
"""
Training script for LSTM models predicting percentage returns.

Instead of predicting absolute prices, these models predict:
    return_t+1 = (price_t+1 / price_t) - 1

Benefits:
- Scale invariant (works for $30 or $500 stocks)
- More stable training (returns typically -5% to +5%)
- Better generalization across stocks
- No data drift issues from price changes

Usage:
    python scripts/train_returns_model.py --symbols AAPL GOOGL --epochs 100
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from loguru import logger
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import StockDataLoader


class ReturnsPreprocessor:
    """Preprocessor for return-based predictions."""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.feature_scaler = None
        self.fitted = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features including returns."""
        df = df.copy()
        
        # Calculate returns
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        df['return_20d'] = df['close'].pct_change(20)
        
        # Volume returns
        df['volume_change'] = df['volume'].pct_change()
        
        # Moving averages of returns
        df['return_ma_5'] = df['return_1d'].rolling(5).mean()
        df['return_ma_20'] = df['return_1d'].rolling(20).mean()
        
        # Volatility of returns
        df['return_std_5'] = df['return_1d'].rolling(5).std()
        df['return_std_20'] = df['return_1d'].rolling(20).std()
        
        # RSI-like momentum
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'] / 100  # Normalize to 0-1
        
        # Price relative to moving averages
        df['price_ma_ratio_7'] = df['close'] / df['close'].rolling(7).mean()
        df['price_ma_ratio_30'] = df['close'] / df['close'].rolling(30).mean()
        
        # High-Low range normalized
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Target: next day return
        df['target_return'] = df['return_1d'].shift(-1)
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def get_features(self) -> list:
        """Feature columns."""
        return [
            'return_1d', 'return_5d', 'return_20d',
            'volume_change', 'return_ma_5', 'return_ma_20',
            'return_std_5', 'return_std_20', 'rsi',
            'price_ma_ratio_7', 'price_ma_ratio_30', 'hl_range'
        ]
    
    def fit_transform(self, df: pd.DataFrame):
        """Fit and transform data."""
        df = self.prepare_features(df)
        features = self.get_features()
        
        # Clip extreme values
        X_data = df[features].values
        X_data = np.clip(X_data, -10, 10)  # Clip extreme returns
        
        y_data = df['target_return'].values
        y_data = np.clip(y_data, -0.2, 0.2)  # Clip extreme target returns (±20%)
        
        # Create sequences
        X, y = [], []
        for i in range(len(X_data) - self.sequence_length):
            X.append(X_data[i:i + self.sequence_length])
            y.append(y_data[i + self.sequence_length - 1])
        
        self.fitted = True
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), df
    
    def save(self, path: Path):
        """Save preprocessor config."""
        config = {
            'sequence_length': self.sequence_length,
            'features': self.get_features(),
            'type': 'returns'
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load preprocessor config."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(sequence_length=config['sequence_length'])


class ReturnsLSTMModel(nn.Module):
    """LSTM model optimized for return prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_size = hidden_size * self.num_directions
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size),
            nn.Tanh(),
            nn.Linear(lstm_out_size, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.LayerNorm(hidden_size),  # LayerNorm better for returns
            nn.GELU(),  # GELU often better for regression
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Tanh()  # Output bounded to [-1, 1] (will be scaled)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Scale output to reasonable return range (e.g., ±20%)
        output = self.fc(context) * 0.2
        
        return output


class ReturnsLSTMPredictor:
    """Predictor for return-based models."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: str = None,
        bidirectional: bool = True
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Using device: {self.device}")
        
        self.model = ReturnsLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        # Huber loss better for returns (robust to outliers)
        self.criterion = nn.HuberLoss(delta=0.05)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15
    ):
        """Train the model with early stopping."""
        logger.info(f"🚀 Starting training for {epochs} epochs (patience={patience})")
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        no_improve = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_t)
                    val_loss = self.criterion(val_pred, y_val_t).item()
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    no_improve = 0
                    # Save best state
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    no_improve += 1
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
                
                if no_improve >= patience:
                    logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_state)
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}")
        
        logger.info(f"✅ Training completed. Best val loss: {self.best_val_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict returns."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_t).cpu().numpy()
        return predictions.flatten()
    
    def save(self, path: Path):
        """Save model."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'bidirectional': self.bidirectional,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_type': 'returns'  # Important flag
        }
        torch.save(checkpoint, path)
        logger.info(f"💾 Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = None):
        """Load model."""
        checkpoint = torch.load(path, map_location=device or 'cpu')
        
        predictor = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            learning_rate=checkpoint['learning_rate'],
            device=device,
            bidirectional=checkpoint.get('bidirectional', True)
        )
        
        predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.train_losses = checkpoint.get('train_losses', [])
        predictor.val_losses = checkpoint.get('val_losses', [])
        predictor.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return predictor


def evaluate_returns_model(model, X_test, y_test, current_prices=None):
    """Evaluate model with return-specific metrics."""
    predictions = model.predict(X_test)
    
    # Ensure same length
    min_len = min(len(predictions), len(y_test))
    predictions = predictions[:min_len]
    y_test = y_test[:min_len]
    
    # Metrics for returns
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    
    # Directional accuracy (most important for returns)
    direction_actual = y_test > 0
    direction_pred = predictions > 0
    directional_accuracy = np.mean(direction_actual == direction_pred)
    
    # Correlation
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, y_test)[0, 1]
    else:
        correlation = 0.0
    
    # MAPE equivalent (treating returns as percentage errors)
    # Since returns are already percentages, MAPE ~ MAE * 100
    mape = mae * 100
    
    return {
        'mae_returns': float(mae),
        'rmse_returns': float(rmse),
        'directional_accuracy': float(directional_accuracy),
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'mape_equivalent': float(mape)
    }


def train_returns_model(
    symbol: str,
    start_date: str = "2021-01-01",
    end_date: str = None,
    epochs: int = 100,
    patience: int = 15,
    save_dir: Path = None
):
    """Train a returns-based model for a symbol."""
    logger.info(f"\n{'='*50}")
    logger.info(f"🎯 Training Returns Model for {symbol}")
    logger.info(f"{'='*50}")
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if save_dir is None:
        save_dir = Path("models/returns")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = StockDataLoader()
    df = loader.load_stock_data(symbol, start_date, end_date)
    logger.info(f"📊 Loaded {len(df)} records for {symbol}")
    
    # Preprocess
    preprocessor = ReturnsPreprocessor(sequence_length=60)
    X, y, df_processed = preprocessor.fit_transform(df)
    logger.info(f"📐 Created {len(X)} sequences with {X.shape[2]} features")
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    logger.info(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model
    model = ReturnsLSTMPredictor(
        input_size=X.shape[2],
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )
    
    model.fit(X_train, y_train, X_val, y_val, epochs=epochs, patience=patience)
    
    # Evaluate
    metrics = evaluate_returns_model(model, X_test, y_test)
    
    logger.info(f"\n📊 Evaluation Results for {symbol}:")
    logger.info(f"   MAE (returns): {metrics['mae_returns']:.4f}")
    logger.info(f"   RMSE (returns): {metrics['rmse_returns']:.4f}")
    logger.info(f"   Directional Accuracy: {metrics['directional_accuracy']*100:.1f}%")
    logger.info(f"   Correlation: {metrics['correlation']:.3f}")
    logger.info(f"   MAPE equivalent: {metrics['mape_equivalent']:.2f}%")
    
    # Save model and preprocessor
    model_path = save_dir / f"lstm_returns_{symbol}.pth"
    config_path = save_dir / f"config_returns_{symbol}.json"
    
    model.save(model_path)
    preprocessor.save(config_path)
    
    # Save metadata
    metadata = {
        'symbol': symbol,
        'model_type': 'returns',
        'trained_at': datetime.now().isoformat(),
        'data_range': f"{start_date} to {end_date}",
        'samples': len(X),
        'metrics': metrics,
        'architecture': {
            'type': 'LSTM',
            'hidden_size': 64,
            'num_layers': 2,
            'bidirectional': True,
            'attention': True
        }
    }
    
    metadata_path = save_dir / f"metadata_returns_{symbol}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Model saved to {model_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train LSTM models for return prediction")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "GOOGL", "MSFT"],
        help="Stock symbols to train"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--start-date", type=str, default="2021-01-01", help="Start date")
    
    args = parser.parse_args()
    
    logger.info("🚀 Returns Model Training Pipeline")
    logger.info(f"   Symbols: {args.symbols}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Patience: {args.patience}")
    
    results = {}
    
    for symbol in args.symbols:
        try:
            metrics = train_returns_model(
                symbol=symbol,
                start_date=args.start_date,
                epochs=args.epochs,
                patience=args.patience
            )
            results[symbol] = metrics
        except Exception as e:
            logger.error(f"❌ Error training {symbol}: {e}")
            results[symbol] = {"error": str(e)}
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TRAINING SUMMARY - Returns Models")
    logger.info("="*60)
    
    for symbol, metrics in results.items():
        if "error" in metrics:
            logger.info(f"❌ {symbol}: {metrics['error']}")
        else:
            logger.info(f"✅ {symbol}: Dir.Acc={metrics['directional_accuracy']*100:.1f}%, MAPE~{metrics['mape_equivalent']:.1f}%")


if __name__ == "__main__":
    main()

