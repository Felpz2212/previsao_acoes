"""
Smart Trainer - Treinamento com modelo base genérico e fine-tuning específico.

Estratégia:
1. Modelo Base: Treinado com múltiplas ações para aprender padrões gerais do mercado
2. Fine-tuning: Ajuste fino para uma ação específica quando solicitado
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import json
import torch

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
    MODELS_DIR
)


# Ações populares para treinar o modelo base
DEFAULT_BASE_STOCKS = [
    'AAPL',   # Apple
    'GOOGL',  # Google
    'MSFT',   # Microsoft
    'AMZN',   # Amazon
    'META',   # Meta (Facebook)
    'NVDA',   # NVIDIA
    'TSLA',   # Tesla
    'JPM',    # JP Morgan
    'V',      # Visa
    'JNJ',    # Johnson & Johnson
]

# Caminhos dos modelos
BASE_MODEL_PATH = MODELS_DIR / "lstm_model_BASE.pth"
BASE_SCALER_PATH = MODELS_DIR / "scaler_BASE.pkl"
BASE_METADATA_PATH = MODELS_DIR / "metadata_BASE.json"


class SmartTrainer:
    """
    Treinador inteligente com duas fases:
    1. Modelo Base: Aprende padrões gerais de múltiplas ações
    2. Fine-tuning: Especializa para uma ação específica
    """
    
    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        learning_rate: float = LSTM_LEARNING_RATE,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT
    ):
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
        
    def train_base_model(
        self,
        symbols: List[str] = None,
        start_date: str = "2019-01-01",
        end_date: str = None,
        epochs: int = None
    ) -> Dict[str, float]:
        """
        Treina modelo base com múltiplas ações para aprender padrões gerais.
        
        Args:
            symbols: Lista de símbolos para treinar (default: DEFAULT_BASE_STOCKS)
            start_date: Data inicial
            end_date: Data final (default: hoje)
            epochs: Número de épocas (default: self.epochs)
            
        Returns:
            Métricas de avaliação
        """
        symbols = symbols or DEFAULT_BASE_STOCKS
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        epochs = epochs or self.epochs
        
        logger.info(f"🎓 Iniciando treinamento do MODELO BASE com {len(symbols)} ações")
        logger.info(f"📊 Ações: {', '.join(symbols)}")
        
        # Coletar dados de todas as ações
        all_data = []
        successful_symbols = []
        
        for symbol in symbols:
            try:
                logger.info(f"📥 Baixando dados de {symbol}...")
                df = self.data_loader.load_stock_data(symbol, start_date, end_date)
                self.data_loader.validate_data(df)
                all_data.append(df)
                successful_symbols.append(symbol)
                logger.info(f"✅ {symbol}: {len(df)} registros")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar {symbol}: {e}")
                continue
        
        if len(all_data) < 3:
            raise ValueError(f"Necessário pelo menos 3 ações, apenas {len(all_data)} carregadas")
        
        logger.info(f"📊 Total de ações carregadas: {len(successful_symbols)}")
        
        # Combinar todos os dados
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"📊 Total de registros combinados: {len(combined_df)}")
        
        # Preprocessar dados
        self.preprocessor = StockDataPreprocessor(sequence_length=self.sequence_length)
        X, y, _ = self.preprocessor.fit_transform(combined_df)
        
        logger.info(f"🔢 Sequências criadas: {len(X)}")
        
        # Split treino/validação/teste
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Inicializar e treinar modelo
        input_size = X_train.shape[2]
        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate
        )
        
        logger.info(f"🧠 Iniciando treinamento por {epochs} épocas...")
        self.model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=self.batch_size)
        
        # Avaliar
        metrics = self._evaluate(X_test, y_test)
        
        # Salvar modelo base
        self._save_base_model(successful_symbols, metrics, start_date, end_date)
        
        logger.info("✅ Modelo BASE treinado e salvo!")
        return metrics
    
    def fine_tune_for_symbol(
        self,
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: str = None,
        epochs: int = None,
        force_retrain: bool = False
    ) -> Dict[str, float]:
        """
        Fine-tuning do modelo base para uma ação específica.
        
        Args:
            symbol: Símbolo da ação
            start_date: Data inicial
            end_date: Data final
            epochs: Épocas de fine-tuning (default: 20% do epochs base)
            force_retrain: Forçar retreino mesmo se modelo existir
            
        Returns:
            Métricas de avaliação
        """
        symbol = symbol.upper()
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        epochs = epochs or max(10, self.epochs // 5)  # 20% das épocas base
        
        model_path = MODELS_DIR / f"lstm_model_{symbol}.pth"
        scaler_path = MODELS_DIR / f"scaler_{symbol}.pkl"
        
        # Verificar se já existe modelo específico
        if model_path.exists() and not force_retrain:
            logger.info(f"✅ Modelo para {symbol} já existe!")
            return self._load_existing_metrics(symbol)
        
        logger.info(f"🎯 Iniciando FINE-TUNING para {symbol}")
        
        # Carregar modelo base
        if not self._load_base_model():
            logger.warning("⚠️ Modelo base não encontrado. Treinando do zero...")
            return self._train_from_scratch(symbol, start_date, end_date, self.epochs)
        
        # Carregar dados da ação específica
        try:
            df = self.data_loader.load_stock_data(symbol, start_date, end_date)
            self.data_loader.validate_data(df)
        except Exception as e:
            raise ValueError(f"❌ Não foi possível carregar dados de {symbol}: {e}")
        
        logger.info(f"📊 {symbol}: {len(df)} registros carregados")
        
        # Criar novo preprocessor para esta ação específica
        specific_preprocessor = StockDataPreprocessor(sequence_length=self.sequence_length)
        X, y, _ = specific_preprocessor.fit_transform(df)
        
        # Split
        train_size = int(0.8 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        # Fine-tuning com learning rate menor
        fine_tune_lr = self.learning_rate * 0.1  # 10% do LR original
        self.model.optimizer = torch.optim.Adam(
            self.model.model.parameters(),
            lr=fine_tune_lr
        )
        
        logger.info(f"🔧 Fine-tuning por {epochs} épocas (LR: {fine_tune_lr})...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size)
        
        # Avaliar com o preprocessor específico
        self.preprocessor = specific_preprocessor
        metrics = self._evaluate(X_test, y_test)
        
        # Salvar modelo específico
        self._save_specific_model(symbol, specific_preprocessor, metrics, start_date, end_date)
        
        logger.info(f"✅ Modelo para {symbol} salvo!")
        return metrics
    
    def get_or_train_model(
        self,
        symbol: str,
        start_date: str = "2020-01-01",
        end_date: str = None
    ) -> Tuple[LSTMPredictor, StockDataPreprocessor]:
        """
        Obtém modelo para uma ação: usa existente, faz fine-tune, ou treina do zero.
        
        Args:
            symbol: Símbolo da ação
            start_date: Data inicial para treino
            end_date: Data final
            
        Returns:
            Tuple (modelo, preprocessor)
        """
        symbol = symbol.upper()
        model_path = MODELS_DIR / f"lstm_model_{symbol}.pth"
        scaler_path = MODELS_DIR / f"scaler_{symbol}.pkl"
        
        # 1. Tentar carregar modelo específico existente
        if model_path.exists() and scaler_path.exists():
            logger.info(f"✅ Carregando modelo existente para {symbol}")
            model = LSTMPredictor.load(model_path)
            preprocessor = StockDataPreprocessor.load(scaler_path)
            return model, preprocessor
        
        # 2. Tentar fine-tuning do modelo base
        if BASE_MODEL_PATH.exists():
            logger.info(f"🔧 Fazendo fine-tuning do modelo base para {symbol}")
            self.fine_tune_for_symbol(symbol, start_date, end_date)
            model = LSTMPredictor.load(model_path)
            preprocessor = StockDataPreprocessor.load(scaler_path)
            return model, preprocessor
        
        # 3. Treinar do zero
        logger.info(f"🆕 Treinando modelo do zero para {symbol}")
        self._train_from_scratch(symbol, start_date, end_date, self.epochs)
        model = LSTMPredictor.load(model_path)
        preprocessor = StockDataPreprocessor.load(scaler_path)
        return model, preprocessor
    
    def _train_from_scratch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        epochs: int
    ) -> Dict[str, float]:
        """Treina modelo do zero para uma ação específica."""
        from src.training.trainer import ModelTrainer
        
        trainer = ModelTrainer(
            symbol=symbol,
            sequence_length=self.sequence_length,
            epochs=epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        return trainer.run_training_pipeline(start_date, end_date)
    
    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Avalia o modelo com métricas."""
        predictions_scaled = self.model.predict(X_test)
        
        # Desnormalizar
        predictions = np.array([
            self.preprocessor.inverse_transform_target(p) 
            for p in predictions_scaled
        ])
        y_real = np.array([
            self.preprocessor.inverse_transform_target(y) 
            for y in y_test
        ])
        
        # Métricas
        mse = np.mean((predictions - y_real) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_real))
        
        mask = y_real != 0
        mape = np.mean(np.abs((y_real[mask] - predictions[mask]) / y_real[mask])) * 100
        
        ss_res = np.sum((y_real - predictions) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        direction_actual = np.diff(y_real) > 0
        direction_pred = np.diff(predictions) > 0
        dir_acc = np.mean(direction_actual == direction_pred) * 100
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'directional_accuracy': float(dir_acc),
            'test_samples': len(y_test)
        }
        
        logger.info(f"📊 Métricas: RMSE=${rmse:.2f}, MAE=${mae:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}")
        return metrics
    
    def _load_base_model(self) -> bool:
        """Carrega o modelo base se existir."""
        if BASE_MODEL_PATH.exists() and BASE_SCALER_PATH.exists():
            try:
                self.model = LSTMPredictor.load(BASE_MODEL_PATH)
                self.preprocessor = StockDataPreprocessor.load(BASE_SCALER_PATH)
                logger.info("✅ Modelo BASE carregado")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar modelo base: {e}")
                return False
        return False
    
    def _save_base_model(
        self,
        symbols: List[str],
        metrics: Dict,
        start_date: str,
        end_date: str
    ):
        """Salva o modelo base."""
        self.model.save(BASE_MODEL_PATH)
        self.preprocessor.save(BASE_SCALER_PATH)
        
        metadata = {
            'type': 'BASE',
            'symbols': symbols,
            'trained_at': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'metrics': metrics,
            'hyperparameters': {
                'sequence_length': self.sequence_length,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }
        
        with open(BASE_METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_specific_model(
        self,
        symbol: str,
        preprocessor: StockDataPreprocessor,
        metrics: Dict,
        start_date: str,
        end_date: str
    ):
        """Salva modelo específico para uma ação."""
        model_path = MODELS_DIR / f"lstm_model_{symbol}.pth"
        scaler_path = MODELS_DIR / f"scaler_{symbol}.pkl"
        metadata_path = MODELS_DIR / f"metadata_{symbol}.json"
        
        self.model.save(model_path)
        preprocessor.save(scaler_path)
        
        metadata = {
            'type': 'FINE_TUNED',
            'symbol': symbol,
            'trained_at': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'metrics': metrics,
            'base_model': 'lstm_model_BASE.pth'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_existing_metrics(self, symbol: str) -> Dict[str, float]:
        """Carrega métricas de um modelo existente."""
        metadata_path = MODELS_DIR / f"metadata_{symbol}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                return data.get('metrics', {})
        return {}


def train_base_model(symbols: List[str] = None, epochs: int = 50) -> Dict[str, float]:
    """Função de conveniência para treinar modelo base."""
    trainer = SmartTrainer(epochs=epochs)
    return trainer.train_base_model(symbols=symbols, epochs=epochs)


def get_model_for_symbol(symbol: str) -> Tuple[LSTMPredictor, StockDataPreprocessor]:
    """Função de conveniência para obter modelo para uma ação."""
    trainer = SmartTrainer()
    return trainer.get_or_train_model(symbol)


if __name__ == "__main__":
    # Teste: treinar modelo base
    logger.info("🚀 Treinando modelo base...")
    
    # Usar menos ações para teste rápido
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    trainer = SmartTrainer(epochs=30)
    metrics = trainer.train_base_model(symbols=test_symbols, epochs=30)
    
    print("\n📊 Métricas do Modelo Base:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

