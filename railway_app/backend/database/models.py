"""
Database Models - SQLAlchemy ORM para PostgreSQL
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class StockPrice(Base):
    """Tabela de preços históricos de ações."""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Features calculadas (opcional, para cache)
    ma_7 = Column(Float, nullable=True)
    ma_30 = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Index composto para queries eficientes
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp', unique=True),
    )
    
    def __repr__(self):
        return f"<StockPrice {self.symbol} {self.timestamp}: ${self.close:.2f}>"


class Prediction(Base):
    """Tabela de previsões realizadas."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Preços
    current_price = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)  # Preenchido depois
    
    # Métricas
    change_percent = Column(Float, nullable=False)
    direction = Column(String(20), nullable=False)  # UP, DOWN, LATERAL
    confidence = Column(String(20), nullable=True)
    
    # Modelo usado
    model_type = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    
    # Timestamps
    prediction_date = Column(DateTime, nullable=False)  # Data da previsão
    target_date = Column(DateTime, nullable=False)  # Data alvo (previsto)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Acurácia (calculada depois)
    error_percent = Column(Float, nullable=True)
    was_correct_direction = Column(Boolean, nullable=True)
    
    def __repr__(self):
        return f"<Prediction {self.symbol} ${self.predicted_price:.2f} ({self.change_percent:+.2f}%)>"


class ModelMetrics(Base):
    """Tabela de métricas de performance dos modelos."""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # BASE, SPECIFIC
    
    # Métricas de treinamento
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    directional_accuracy = Column(Float, nullable=True)
    
    # Info de treinamento
    train_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)
    epochs = Column(Integer, nullable=True)
    
    # Período de dados
    data_start = Column(DateTime, nullable=True)
    data_end = Column(DateTime, nullable=True)
    
    trained_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelMetrics {self.symbol} RMSE={self.rmse:.2f}>"


class TrainingLog(Base):
    """Log de treinamentos realizados."""
    __tablename__ = 'training_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Status
    status = Column(String(20), nullable=False)  # STARTED, COMPLETED, FAILED
    error_message = Column(Text, nullable=True)
    
    # Configuração
    epochs = Column(Integer, nullable=True)
    batch_size = Column(Integer, nullable=True)
    learning_rate = Column(Float, nullable=True)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<TrainingLog {self.symbol} {self.status}>"

