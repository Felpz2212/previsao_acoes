"""
Model Evaluation Service - Avalia performance do modelo em produção.

Funcionalidades:
- Compara previsões passadas com valores reais
- Calcula métricas de acurácia (MAPE, acurácia direcional)
- Detecta drift nos dados e nas previsões
- Gera alertas quando performance degrada
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from sqlalchemy import text


@dataclass
class PredictionEvaluation:
    """Avaliação de uma previsão individual."""
    symbol: str
    prediction_date: datetime
    predicted_price: float
    actual_price: float
    current_price_at_prediction: float
    error_percent: float
    predicted_direction: str  # UP, DOWN
    actual_direction: str
    direction_correct: bool


class ModelEvaluationService:
    """Serviço para avaliar modelo em produção."""
    
    def __init__(self, db_service):
        self.db = db_service
        logger.info("📊 ModelEvaluationService inicializado")
    
    def evaluate_past_predictions(self, days: int = 7) -> Dict:
        """
        Avalia previsões passadas comparando com valores reais.
        
        Args:
            days: Número de dias para avaliar
            
        Returns:
            Métricas de performance do modelo
        """
        if not self.db:
            return {"error": "Database não disponível"}
        
        session = self.db.get_session()
        
        try:
            # Buscar previsões dos últimos N dias
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            result = session.execute(text("""
                SELECT 
                    p.symbol,
                    p.prediction_date,
                    p.predicted_price,
                    p.current_price,
                    p.direction,
                    p.created_at
                FROM predictions p
                WHERE p.created_at >= :cutoff
                ORDER BY p.created_at DESC
            """), {"cutoff": cutoff_date})
            
            predictions = result.fetchall()
            
            if not predictions:
                return {
                    "status": "no_data",
                    "message": f"Nenhuma previsão encontrada nos últimos {days} dias"
                }
            
            evaluations = []
            
            for pred in predictions:
                symbol = pred[0]
                prediction_date = pred[1]
                predicted_price = pred[2]
                current_price_at_pred = pred[3]
                predicted_direction = pred[4]
                created_at = pred[5]
                
                # Buscar preço real do dia seguinte à previsão
                next_day = created_at + timedelta(days=1)
                
                actual_result = session.execute(text("""
                    SELECT close 
                    FROM stock_prices 
                    WHERE symbol = :symbol 
                    AND timestamp >= :start_date
                    AND timestamp < :end_date
                    ORDER BY timestamp ASC
                    LIMIT 1
                """), {
                    "symbol": symbol,
                    "start_date": next_day.date(),
                    "end_date": (next_day + timedelta(days=1)).date()
                })
                
                actual_row = actual_result.fetchone()
                
                if actual_row:
                    actual_price = float(actual_row[0])
                    
                    # Calcular erro
                    error_percent = abs(predicted_price - actual_price) / actual_price * 100
                    
                    # Direção real
                    actual_direction = "ALTA" if actual_price > current_price_at_pred else "BAIXA"
                    
                    # Direção prevista estava correta?
                    pred_dir_up = "ALTA" in predicted_direction.upper()
                    actual_dir_up = actual_direction == "ALTA"
                    direction_correct = pred_dir_up == actual_dir_up
                    
                    evaluations.append(PredictionEvaluation(
                        symbol=symbol,
                        prediction_date=created_at,
                        predicted_price=predicted_price,
                        actual_price=actual_price,
                        current_price_at_prediction=current_price_at_pred,
                        error_percent=error_percent,
                        predicted_direction=predicted_direction,
                        actual_direction=actual_direction,
                        direction_correct=direction_correct
                    ))
            
            if not evaluations:
                return {
                    "status": "no_verified",
                    "message": "Previsões encontradas, mas ainda não há dados reais para comparar",
                    "predictions_pending": len(predictions)
                }
            
            # Calcular métricas agregadas
            errors = [e.error_percent for e in evaluations]
            directions = [e.direction_correct for e in evaluations]
            
            mape = np.mean(errors)
            direction_accuracy = np.mean(directions) * 100
            
            # Por símbolo
            by_symbol = {}
            symbols = set(e.symbol for e in evaluations)
            for symbol in symbols:
                symbol_evals = [e for e in evaluations if e.symbol == symbol]
                by_symbol[symbol] = {
                    "count": len(symbol_evals),
                    "mape": round(np.mean([e.error_percent for e in symbol_evals]), 2),
                    "direction_accuracy": round(np.mean([e.direction_correct for e in symbol_evals]) * 100, 1),
                    "avg_error_dollars": round(np.mean([abs(e.predicted_price - e.actual_price) for e in symbol_evals]), 2)
                }
            
            return {
                "status": "success",
                "period_days": days,
                "total_evaluated": len(evaluations),
                "metrics": {
                    "mape_percent": round(mape, 2),
                    "direction_accuracy_percent": round(direction_accuracy, 1),
                    "avg_error_dollars": round(np.mean([abs(e.predicted_price - e.actual_price) for e in evaluations]), 2),
                    "max_error_percent": round(max(errors), 2),
                    "min_error_percent": round(min(errors), 2)
                },
                "by_symbol": by_symbol,
                "quality_assessment": self._assess_quality(mape, direction_accuracy),
                "recent_evaluations": [
                    {
                        "symbol": e.symbol,
                        "date": e.prediction_date.isoformat() if e.prediction_date else None,
                        "predicted": round(e.predicted_price, 2),
                        "actual": round(e.actual_price, 2),
                        "error_percent": round(e.error_percent, 2),
                        "direction_correct": e.direction_correct
                    }
                    for e in evaluations[:10]
                ]
            }
            
        except Exception as e:
            logger.error(f"Erro ao avaliar previsões: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            session.close()
    
    def _assess_quality(self, mape: float, direction_accuracy: float) -> Dict:
        """Avalia qualidade do modelo com base nas métricas."""
        
        # MAPE assessment
        if mape < 3:
            mape_grade = "EXCELENTE"
            mape_status = "green"
        elif mape < 5:
            mape_grade = "BOM"
            mape_status = "green"
        elif mape < 10:
            mape_grade = "ACEITAVEL"
            mape_status = "yellow"
        else:
            mape_grade = "PRECISA_MELHORIA"
            mape_status = "red"
        
        # Direction assessment
        if direction_accuracy >= 60:
            dir_grade = "EXCELENTE"
            dir_status = "green"
        elif direction_accuracy >= 55:
            dir_grade = "BOM"
            dir_status = "green"
        elif direction_accuracy >= 50:
            dir_grade = "ACEITAVEL"
            dir_status = "yellow"
        else:
            dir_grade = "ABAIXO_RANDOM"
            dir_status = "red"
        
        # Overall
        if mape_status == "green" and dir_status == "green":
            overall = "SAUDAVEL"
        elif mape_status == "red" or dir_status == "red":
            overall = "PRECISA_ATENCAO"
        else:
            overall = "MONITORAR"
        
        return {
            "overall": overall,
            "mape": {"grade": mape_grade, "status": mape_status},
            "direction": {"grade": dir_grade, "status": dir_status},
            "recommendation": self._get_recommendation(mape, direction_accuracy)
        }
    
    def _get_recommendation(self, mape: float, direction_accuracy: float) -> str:
        """Gera recomendação baseada nas métricas."""
        if mape > 10 and direction_accuracy < 50:
            return "URGENTE: Retreinar modelo com dados mais recentes"
        elif mape > 10:
            return "Considerar retreinar modelo - erro alto"
        elif direction_accuracy < 50:
            return "Acurácia direcional baixa - revisar features"
        elif mape > 5:
            return "Performance aceitável - monitorar tendências"
        else:
            return "Modelo saudável - manter monitoramento"
    
    def detect_data_drift(self, symbol: str, window_days: int = 30) -> Dict:
        """
        Detecta drift nos dados de entrada.
        
        Compara estatísticas dos dados recentes com histórico.
        """
        if not self.db:
            return {"error": "Database não disponível"}
        
        session = self.db.get_session()
        
        try:
            # Dados recentes (última semana)
            recent_start = datetime.utcnow() - timedelta(days=7)
            
            # Dados históricos (últimos N dias exceto última semana)
            historical_start = datetime.utcnow() - timedelta(days=window_days)
            
            # Stats recentes
            recent_result = session.execute(text("""
                SELECT 
                    AVG(close) as avg_price,
                    STDDEV(close) as std_price,
                    AVG(volume) as avg_volume,
                    STDDEV(volume) as std_volume
                FROM stock_prices
                WHERE symbol = :symbol
                AND timestamp >= :start_date
            """), {"symbol": symbol, "start_date": recent_start})
            
            recent_stats = recent_result.fetchone()
            
            # Stats históricos
            historical_result = session.execute(text("""
                SELECT 
                    AVG(close) as avg_price,
                    STDDEV(close) as std_price,
                    AVG(volume) as avg_volume,
                    STDDEV(volume) as std_volume
                FROM stock_prices
                WHERE symbol = :symbol
                AND timestamp >= :start_date
                AND timestamp < :end_date
            """), {"symbol": symbol, "start_date": historical_start, "end_date": recent_start})
            
            historical_stats = historical_result.fetchone()
            
            if not recent_stats[0] or not historical_stats[0]:
                return {"status": "insufficient_data", "symbol": symbol}
            
            # Calcular drift
            price_drift = abs(recent_stats[0] - historical_stats[0]) / historical_stats[0] * 100 if historical_stats[0] else 0
            volatility_change = abs(recent_stats[1] - historical_stats[1]) / historical_stats[1] * 100 if historical_stats[1] else 0
            volume_drift = abs(recent_stats[2] - historical_stats[2]) / historical_stats[2] * 100 if historical_stats[2] else 0
            
            # Avaliar drift
            drift_detected = price_drift > 10 or volatility_change > 50 or volume_drift > 100
            
            return {
                "symbol": symbol,
                "drift_detected": drift_detected,
                "metrics": {
                    "price_drift_percent": round(price_drift, 2),
                    "volatility_change_percent": round(volatility_change, 2),
                    "volume_drift_percent": round(volume_drift, 2)
                },
                "recent_period": f"last 7 days",
                "historical_period": f"last {window_days} days",
                "recommendation": "Considerar retreinar modelo" if drift_detected else "Dados estáveis"
            }
            
        except Exception as e:
            logger.error(f"Erro ao detectar drift: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            session.close()


# Singleton
_evaluation_service: Optional[ModelEvaluationService] = None


def get_evaluation_service(db_service) -> ModelEvaluationService:
    """Obtém instância do serviço de avaliação."""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = ModelEvaluationService(db_service)
    return _evaluation_service

