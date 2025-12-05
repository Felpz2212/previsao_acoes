"""
Improved Training Pipeline - Treinamento robusto com validacao avançada.

Melhorias:
- Early Stopping para evitar overfitting
- Learning Rate Scheduler (ReduceLROnPlateau)
- Gradient Clipping para estabilidade
- Walk-forward validation (split temporal correto)
- Mais metricas e logging
- Salvamento do melhor modelo
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger

from src.data.data_loader import StockDataLoader
from src.data.preprocessor import StockDataPreprocessor
from src.models.lstm_model import LSTMModel
from config.settings import MODELS_DIR


class ImprovedLSTMPredictor:
    """
    LSTM Predictor com melhorias para producao.
    
    Features:
    - Early stopping automatico
    - Learning rate scheduler
    - Gradient clipping
    - Best model checkpointing
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,  # Aumentado de 50
        num_layers: int = 3,    # Aumentado de 2
        dropout: float = 0.3,   # Aumentado de 0.2
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        bidirectional: bool = True  # LSTM bidirecional
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
        
        logger.info(f"🖥️ Usando device: {self.device}")
        
        # Modelo melhorado
        self.model = self._build_model().to(self.device)
        
        # Loss e optimizer
        self.criterion = nn.HuberLoss(delta=1.0)  # Mais robusto que MSE
        self.optimizer = torch.optim.AdamW(  # AdamW com weight decay
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def _build_model(self) -> nn.Module:
        """Constroi o modelo LSTM melhorado."""
        
        class ImprovedLSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
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
                
                # Attention layer (opcional, melhora previsoes)
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
        
        return ImprovedLSTMModel(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.bidirectional
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,  # Early stopping patience
        gradient_clip: float = 1.0,  # Gradient clipping
        verbose: bool = True
    ) -> Dict:
        """
        Treina o modelo com early stopping e validação.
        
        Args:
            X_train: Features de treino (N, seq_len, features)
            y_train: Targets de treino (N,)
            X_val: Features de validação
            y_val: Targets de validação
            epochs: Máximo de épocas
            batch_size: Tamanho do batch
            patience: Épocas sem melhoria antes de parar
            gradient_clip: Max norm para gradient clipping
            verbose: Mostrar progresso
            
        Returns:
            Dict com métricas de treinamento
        """
        logger.info(f"🚀 Iniciando treinamento: {epochs} épocas, batch={batch_size}")
        logger.info(f"📊 Dados: {len(X_train)} treino, {len(X_val) if X_val is not None else 0} validação")
        
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # ====== TREINO ======
            self.model.train()
            train_loss = self._train_epoch(X_train, y_train, batch_size, gradient_clip)
            self.train_losses.append(train_loss)
            
            # ====== VALIDAÇÃO ======
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_loss = self._validate(X_val, y_val, batch_size)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Check best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    epochs_without_improvement = 0
                    
                    if verbose:
                        logger.info(f"✅ Epoch {epoch+1}: Nova melhor val_loss={val_loss:.6f}")
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    logger.info(f"⏹️ Early stopping na época {epoch+1}")
                    break
                
                if verbose and (epoch + 1) % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                        f"LR: {current_lr:.2e} | Patience: {epochs_without_improvement}/{patience}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}")
        
        # Restaurar melhor modelo
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"✅ Modelo restaurado para melhor checkpoint (val_loss={self.best_val_loss:.6f})")
        
        return {
            'final_train_loss': self.train_losses[-1],
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses),
            'early_stopped': epochs_without_improvement >= patience
        }
    
    def _train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        gradient_clip: float
    ) -> float:
        """Treina uma época."""
        total_loss = 0
        num_batches = 0
        
        # Shuffle indices
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i + batch_size]
            
            X_batch = torch.FloatTensor(X_train[batch_idx]).to(self.device)
            y_batch = torch.FloatTensor(y_train[batch_idx]).reshape(-1, 1).to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward com gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int) -> float:
        """Valida o modelo."""
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = torch.FloatTensor(X_val[i:i + batch_size]).to(self.device)
                y_batch = torch.FloatTensor(y_val[i:i + batch_size]).reshape(-1, 1).to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
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
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        
        logger.info(f"✅ Modelo carregado: {path}")
        return predictor


class ImprovedModelTrainer:
    """
    Trainer melhorado com validação robusta.
    
    Features:
    - Walk-forward validation (temporal split correto)
    - Múltiplas métricas de avaliação
    - Salvamento automático do melhor modelo
    - Logging detalhado
    """
    
    def __init__(
        self,
        symbol: str,
        sequence_length: int = 60,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        patience: int = 15
    ):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.patience = patience
        
        self.data_loader = StockDataLoader()
        self.preprocessor = None
        self.model = None
        self.metrics = {}
    
    def run_training_pipeline(
        self,
        start_date: str,
        end_date: str,
        train_split: float = 0.7,
        val_split: float = 0.15
    ) -> Dict:
        """
        Pipeline completo de treinamento.
        
        Split temporal:
        - 70% treino (dados mais antigos)
        - 15% validação
        - 15% teste (dados mais recentes)
        """
        logger.info(f"=" * 60)
        logger.info(f"🎯 Treinando modelo para {self.symbol}")
        logger.info(f"📅 Período: {start_date} até {end_date}")
        logger.info(f"=" * 60)
        
        # 1. Carregar dados
        logger.info("📥 Carregando dados...")
        df = self.data_loader.load_stock_data(self.symbol, start_date, end_date)
        self.data_loader.validate_data(df)
        logger.info(f"✅ {len(df)} registros carregados")
        
        # 2. Preprocessar
        logger.info("⚙️ Preprocessando dados...")
        self.preprocessor = StockDataPreprocessor(sequence_length=self.sequence_length)
        X, y, df_processed = self.preprocessor.fit_transform(df)
        logger.info(f"✅ {len(X)} sequências criadas com {X.shape[2]} features")
        
        # 3. Split temporal (IMPORTANTE: sem shuffle!)
        n = len(X)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"📊 Split temporal:")
        logger.info(f"   Treino: {len(X_train)} ({train_split*100:.0f}%)")
        logger.info(f"   Validação: {len(X_val)} ({val_split*100:.0f}%)")
        logger.info(f"   Teste: {len(X_test)} ({(1-train_split-val_split)*100:.0f}%)")
        
        # 4. Treinar modelo
        logger.info("🧠 Treinando modelo...")
        self.model = ImprovedLSTMPredictor(
            input_size=X_train.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate
        )
        
        train_results = self.model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            patience=self.patience
        )
        
        # 5. Avaliar no conjunto de teste
        logger.info("📊 Avaliando modelo...")
        self.metrics = self._evaluate(X_test, y_test)
        self.metrics.update(train_results)
        
        # 6. Salvar modelo
        logger.info("💾 Salvando modelo...")
        self._save_model()
        
        # 7. Resumo final
        self._print_summary()
        
        return self.metrics
    
    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Avalia modelo com múltiplas métricas."""
        
        # Previsões (normalizadas)
        preds_scaled = self.model.predict(X_test)
        
        # Inverse transform para valores reais
        predictions = np.array([
            self.preprocessor.inverse_transform_target(p) for p in preds_scaled
        ])
        actuals = np.array([
            self.preprocessor.inverse_transform_target(y) for y in y_test
        ])
        
        # Métricas
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        # MAPE (evitando divisão por zero)
        mask = actuals != 0
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        
        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Acurácia direcional
        dir_actual = np.diff(actuals) > 0
        dir_pred = np.diff(predictions) > 0
        directional_accuracy = np.mean(dir_actual == dir_pred) * 100
        
        # Erro percentual médio
        mean_percent_error = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'directional_accuracy': float(directional_accuracy),
            'mean_percent_error': float(mean_percent_error),
            'test_samples': len(y_test),
            'predictions_sample': predictions[:5].tolist(),
            'actuals_sample': actuals[:5].tolist()
        }
    
    def _save_model(self):
        """Salva modelo e metadata."""
        model_path = MODELS_DIR / f"lstm_model_{self.symbol}.pth"
        scaler_path = MODELS_DIR / f"scaler_{self.symbol}.pkl"
        metadata_path = MODELS_DIR / f"metadata_{self.symbol}.json"
        
        # Salvar modelo
        self.model.save(model_path)
        
        # Salvar preprocessor
        self.preprocessor.save(scaler_path)
        
        # Salvar metadata
        metadata = {
            'symbol': self.symbol,
            'trained_at': datetime.now().isoformat(),
            'model_type': 'ImprovedLSTM',
            'architecture': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': True
            },
            'training': {
                'epochs_requested': self.epochs,
                'epochs_trained': self.metrics.get('epochs_trained', 0),
                'early_stopped': self.metrics.get('early_stopped', False),
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'patience': self.patience
            },
            'metrics': self.metrics
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Arquivos salvos em {MODELS_DIR}")
    
    def _print_summary(self):
        """Imprime resumo do treinamento."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 RESUMO DO TREINAMENTO")
        logger.info("=" * 60)
        logger.info(f"  Símbolo: {self.symbol}")
        logger.info(f"  Épocas treinadas: {self.metrics.get('epochs_trained', '?')}")
        logger.info(f"  Early stopping: {self.metrics.get('early_stopped', False)}")
        logger.info("")
        logger.info("📈 MÉTRICAS DE AVALIAÇÃO:")
        logger.info(f"  RMSE: ${self.metrics.get('rmse', 0):.2f}")
        logger.info(f"  MAE: ${self.metrics.get('mae', 0):.2f}")
        logger.info(f"  MAPE: {self.metrics.get('mape', 0):.2f}%")
        logger.info(f"  R²: {self.metrics.get('r2', 0):.4f}")
        logger.info(f"  Acurácia Direcional: {self.metrics.get('directional_accuracy', 0):.2f}%")
        logger.info("")
        logger.info("🎯 PREVISÕES vs REAIS (amostra):")
        preds = self.metrics.get('predictions_sample', [])
        actuals = self.metrics.get('actuals_sample', [])
        for i, (p, a) in enumerate(zip(preds, actuals)):
            error = abs(p - a) / a * 100 if a != 0 else 0
            logger.info(f"  {i+1}. Previsto: ${p:.2f} | Real: ${a:.2f} | Erro: {error:.2f}%")
        logger.info("=" * 60)


if __name__ == "__main__":
    # Teste do trainer melhorado
    trainer = ImprovedModelTrainer(
        symbol="AAPL",
        epochs=100,
        patience=15
    )
    
    metrics = trainer.run_training_pipeline(
        start_date="2018-01-01",
        end_date="2024-12-01"
    )
    
    print("\nMétricas finais:", metrics)

