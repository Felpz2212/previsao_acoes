"""
Database Service - Conexão e operações com PostgreSQL
"""
import os
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
import pandas as pd
from loguru import logger

from database.models import Base, StockPrice, Prediction, ModelMetrics, TrainingLog


class DatabaseService:
    """Serviço de conexão e operações com PostgreSQL."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Inicializa conexão com o banco.
        
        Args:
            database_url: URL de conexão (default: variável de ambiente DATABASE_URL)
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        
        if not self.database_url:
            logger.warning("⚠️ DATABASE_URL não configurada. Usando SQLite local.")
            self.database_url = "sqlite:///./stock_predictor.db"
        
        # Railway usa postgres:// mas SQLAlchemy precisa de postgresql://
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
        
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info(f"🗃️ Database conectado: {self.database_url[:30]}...")
    
    def create_tables(self):
        """Cria todas as tabelas no banco."""
        try:
            logger.info("📝 Criando tabelas no banco de dados...")
            Base.metadata.create_all(self.engine)
            
            # Verificar quais tabelas existem
            from sqlalchemy import inspect
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            logger.info(f"✅ Tabelas no banco: {tables}")
            
            expected = ['stock_prices', 'predictions', 'model_metrics', 'training_logs']
            missing = [t for t in expected if t not in tables]
            if missing:
                logger.warning(f"⚠️ Tabelas faltando: {missing}")
            else:
                logger.info("✅ Todas as tabelas criadas com sucesso!")
                
        except Exception as e:
            logger.error(f"❌ Erro ao criar tabelas: {e}")
            raise
    
    def get_database_status(self) -> dict:
        """Retorna status detalhado do banco de dados."""
        try:
            from sqlalchemy import inspect, text
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            session = self.get_session()
            
            # Contar registros em cada tabela
            table_counts = {}
            for table in tables:
                try:
                    count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    table_counts[table] = count
                except:
                    table_counts[table] = "error"
            
            session.close()
            
            return {
                "status": "connected",
                "tables": tables,
                "record_counts": table_counts,
                "database_url": self.database_url[:30] + "..."
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_session(self) -> Session:
        """Retorna uma nova sessão."""
        return self.SessionLocal()
    
    # ==================== STOCK PRICES ====================
    
    def save_stock_prices(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Salva preços de ações no banco (UPSERT - ignora duplicados).
        
        Args:
            symbol: Ticker da ação
            df: DataFrame com colunas timestamp, open, high, low, close, volume
        
        Returns:
            Número de registros salvos
        """
        session = self.get_session()
        saved = 0
        skipped = 0
        
        try:
            for _, row in df.iterrows():
                # Verificar se já existe
                existing = session.query(StockPrice).filter(
                    StockPrice.symbol == symbol.upper(),
                    StockPrice.timestamp == row['timestamp']
                ).first()
                
                if existing:
                    # Atualizar registro existente
                    existing.open = float(row['open'])
                    existing.high = float(row['high'])
                    existing.low = float(row['low'])
                    existing.close = float(row['close'])
                    existing.volume = float(row['volume'])
                    existing.ma_7 = float(row.get('ma_7', 0)) if pd.notna(row.get('ma_7')) else None
                    existing.ma_30 = float(row.get('ma_30', 0)) if pd.notna(row.get('ma_30')) else None
                    skipped += 1
                else:
                    # Criar novo registro
                    price = StockPrice(
                        symbol=symbol.upper(),
                        timestamp=row['timestamp'],
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']),
                        ma_7=float(row.get('ma_7', 0)) if pd.notna(row.get('ma_7')) else None,
                        ma_30=float(row.get('ma_30', 0)) if pd.notna(row.get('ma_30')) else None,
                    )
                    session.add(price)
                    saved += 1
            
            session.commit()
            logger.info(f"💾 {saved} novos preços salvos para {symbol} ({skipped} atualizados)")
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Erro ao salvar preços: {e}")
            # Não re-raise para não quebrar o fluxo
        finally:
            session.close()
        
        return saved
    
    def get_stock_prices(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Obtém preços de ações do banco.
        
        Returns:
            DataFrame com preços ou DataFrame vazio se não encontrar
        """
        session = self.get_session()
        
        try:
            query = session.query(StockPrice).filter(
                StockPrice.symbol == symbol.upper()
            )
            
            if start_date:
                query = query.filter(StockPrice.timestamp >= start_date)
            if end_date:
                query = query.filter(StockPrice.timestamp <= end_date)
            
            query = query.order_by(StockPrice.timestamp.desc()).limit(limit)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            data = [{
                'timestamp': r.timestamp,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume,
                'ma_7': r.ma_7,
                'ma_30': r.ma_30,
            } for r in results]
            
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"📊 {len(df)} preços carregados do DB para {symbol}")
            return df
            
        finally:
            session.close()
    
    def has_recent_data(self, symbol: str, max_age_hours: int = 24) -> bool:
        """Verifica se há dados recentes no banco."""
        session = self.get_session()
        
        try:
            cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            count = session.query(StockPrice).filter(
                StockPrice.symbol == symbol.upper(),
                StockPrice.timestamp >= cutoff
            ).count()
            
            return count > 0
            
        finally:
            session.close()
    
    # ==================== PREDICTIONS ====================
    
    def save_prediction(
        self,
        symbol: str,
        current_price: float,
        predicted_price: float,
        change_percent: float,
        direction: str,
        model_type: str,
        confidence: str = None,
        target_date: datetime = None
    ) -> int:
        """Salva uma previsão no banco."""
        session = self.get_session()
        
        try:
            prediction = Prediction(
                symbol=symbol.upper(),
                current_price=current_price,
                predicted_price=predicted_price,
                change_percent=change_percent,
                direction=direction,
                confidence=confidence,
                model_type=model_type,
                prediction_date=datetime.utcnow(),
                target_date=target_date or (datetime.utcnow() + timedelta(days=1))
            )
            
            session.add(prediction)
            session.commit()
            
            logger.info(f"💾 Previsão salva: {symbol} ${predicted_price:.2f}")
            return prediction.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Erro ao salvar previsão: {e}")
            raise
        finally:
            session.close()
    
    def get_predictions_history(
        self, 
        symbol: Optional[str] = None, 
        limit: int = 50
    ) -> List[Dict]:
        """Obtém histórico de previsões."""
        session = self.get_session()
        
        try:
            query = session.query(Prediction)
            
            if symbol:
                query = query.filter(Prediction.symbol == symbol.upper())
            
            query = query.order_by(desc(Prediction.created_at)).limit(limit)
            
            results = query.all()
            
            return [{
                'id': r.id,
                'symbol': r.symbol,
                'current_price': r.current_price,
                'predicted_price': r.predicted_price,
                'actual_price': r.actual_price,
                'change_percent': r.change_percent,
                'direction': r.direction,
                'model_type': r.model_type,
                'prediction_date': r.prediction_date.isoformat() if r.prediction_date else None,
                'target_date': r.target_date.isoformat() if r.target_date else None,
                'was_correct_direction': r.was_correct_direction,
                'error_percent': r.error_percent
            } for r in results]
            
        finally:
            session.close()
    
    def update_prediction_actual(self, prediction_id: int, actual_price: float):
        """Atualiza previsão com preço real (para calcular acurácia)."""
        session = self.get_session()
        
        try:
            prediction = session.query(Prediction).get(prediction_id)
            
            if prediction:
                prediction.actual_price = actual_price
                
                # Calcular erro
                prediction.error_percent = abs(
                    (actual_price - prediction.predicted_price) / actual_price
                ) * 100
                
                # Verificar se direção estava correta
                actual_direction = "UP" if actual_price > prediction.current_price else "DOWN"
                predicted_direction = "UP" if prediction.predicted_price > prediction.current_price else "DOWN"
                prediction.was_correct_direction = actual_direction == predicted_direction
                
                session.commit()
                logger.info(f"✅ Previsão {prediction_id} atualizada com preço real")
            
        finally:
            session.close()
    
    # ==================== MODEL METRICS ====================
    
    def save_model_metrics(
        self,
        symbol: str,
        model_type: str,
        metrics: Dict,
        train_samples: int = None,
        test_samples: int = None,
        epochs: int = None,
        data_start: datetime = None,
        data_end: datetime = None
    ):
        """Salva métricas de um modelo treinado."""
        session = self.get_session()
        
        try:
            model_metrics = ModelMetrics(
                symbol=symbol.upper(),
                model_type=model_type,
                rmse=metrics.get('rmse'),
                mae=metrics.get('mae'),
                mape=metrics.get('mape'),
                r2=metrics.get('r2'),
                directional_accuracy=metrics.get('directional_accuracy'),
                train_samples=train_samples,
                test_samples=test_samples,
                epochs=epochs,
                data_start=data_start,
                data_end=data_end
            )
            
            session.add(model_metrics)
            session.commit()
            
            logger.info(f"💾 Métricas salvas para {symbol}: RMSE={metrics.get('rmse', 0):.2f}")
            
        finally:
            session.close()
    
    def get_model_performance(self, symbol: str = None) -> List[Dict]:
        """Obtém histórico de performance dos modelos."""
        session = self.get_session()
        
        try:
            query = session.query(ModelMetrics)
            
            if symbol:
                query = query.filter(ModelMetrics.symbol == symbol.upper())
            
            query = query.order_by(desc(ModelMetrics.trained_at))
            
            results = query.all()
            
            return [{
                'symbol': r.symbol,
                'model_type': r.model_type,
                'rmse': r.rmse,
                'mae': r.mae,
                'mape': r.mape,
                'r2': r.r2,
                'directional_accuracy': r.directional_accuracy,
                'trained_at': r.trained_at.isoformat() if r.trained_at else None
            } for r in results]
            
        finally:
            session.close()


# Instância global (singleton)
_db_service: Optional[DatabaseService] = None


def get_db_service() -> DatabaseService:
    """Obtém instância do DatabaseService (singleton)."""
    global _db_service
    
    if _db_service is None:
        _db_service = DatabaseService()
        _db_service.create_tables()
    
    return _db_service

