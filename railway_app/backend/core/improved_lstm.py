"""
Improved LSTM Model - Arquitetura melhorada com Attention
Para carregamento de modelos treinados com ImprovedLSTMPredictor
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger


class ImprovedLSTMModel(nn.Module):
    """Modelo LSTM bidirecional com Attention."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention layer
        lstm_output_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.Tanh(),
            nn.Linear(lstm_output_size, 1)
        )
        
        # Fully connected layers com BatchNorm
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        
        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Fully connected
        output = self.fc(context)
        
        return output


class ImprovedLSTMPredictor:
    """Predictor para modelos LSTM melhorados."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        bidirectional: bool = True
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        
        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"🖥️ ImprovedLSTM usando device: {self.device}")
        
        # Modelo
        self.model = ImprovedLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        # Training history (para compatibilidade)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz previsões."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions.flatten()
    
    def save(self, path: Path):
        """Salva modelo."""
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
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)
        logger.info(f"💾 Modelo salvo: {path}")
    
    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> 'ImprovedLSTMPredictor':
        """Carrega modelo."""
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
        
        logger.info(f"✅ ImprovedLSTM carregado: {path}")
        return predictor


def detect_model_type(checkpoint_path: Path) -> str:
    """Detecta se o modelo é original ou melhorado."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Se tem 'bidirectional', é modelo melhorado
    if 'bidirectional' in checkpoint:
        return 'improved'
    
    # Se state_dict tem 'attention', é modelo melhorado
    state_dict = checkpoint.get('model_state_dict', {})
    if any('attention' in k for k in state_dict.keys()):
        return 'improved'
    
    return 'original'

