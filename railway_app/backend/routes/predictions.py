"""
Predictions Routes - Previsões LSTM com persistência
"""
from fastapi import APIRouter, HTTPException, Query, Request
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from loguru import logger

from services.stock_service import StockService
from services.model_service import ModelService


router = APIRouter()
stock_service = StockService()


class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    change_percent: float
    direction: str
    confidence: str
    model_type: str
    indicators: dict
    timestamp: str


@router.get("/{symbol}", response_model=PredictionResponse)
async def get_prediction(request: Request, symbol: str):
    """
    Obtém previsão LSTM para uma ação.
    
    - Usa modelo específico se disponível
    - Fallback para modelo BASE
    - Salva previsão no PostgreSQL
    """
    from routes.stocks import resolve_symbol
    resolved_symbol = resolve_symbol(symbol)
    
    logger.info(f"🔮 Requisição de previsão para {resolved_symbol}")
    
    try:
        # Obter dados
        df = stock_service.get_stock_data(resolved_symbol, days=400)
        
        if df is None or len(df) < 70:
            raise HTTPException(
                status_code=400,
                detail=f"Dados insuficientes para {resolved_symbol}. Mínimo: 70 dias, encontrado: {len(df) if df is not None else 0}"
            )
        
        current_price = float(df['close'].iloc[-1])
        
        # Obter modelo e fazer previsão
        model_service: ModelService = request.app.state.model_service
        prediction_result = model_service.predict(resolved_symbol, df)
        
        predicted_price = prediction_result['predicted_price']
        model_type = prediction_result['model_type']
        
        # Calcular métricas
        change_percent = ((predicted_price - current_price) / current_price) * 100
        
        if change_percent > 2:
            direction = "ALTA FORTE"
            confidence = "Alta"
        elif change_percent > 0.5:
            direction = "ALTA"
            confidence = "Moderada"
        elif change_percent < -2:
            direction = "BAIXA FORTE"
            confidence = "Alta"
        elif change_percent < -0.5:
            direction = "BAIXA"
            confidence = "Moderada"
        else:
            direction = "LATERAL"
            confidence = "Baixa"
        
        # Indicadores técnicos
        ma_7 = float(df['close'].rolling(7).mean().iloc[-1])
        ma_30 = float(df['close'].rolling(30).mean().iloc[-1])
        
        indicators = {
            "ma_7": round(ma_7, 2),
            "ma_30": round(ma_30, 2),
            "trend": "bullish" if ma_7 > ma_30 else "bearish"
        }
        
        # Salvar previsão no banco de dados
        db = getattr(request.app.state, 'db', None)
        if db:
            try:
                db.save_prediction(
                    symbol=resolved_symbol,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    change_percent=change_percent,
                    direction=direction,
                    model_type=model_type,
                    confidence=confidence
                )
                logger.info(f"💾 Previsão salva no PostgreSQL")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao salvar previsão no DB: {e}")
        
        response = PredictionResponse(
            symbol=resolved_symbol,
            current_price=round(current_price, 2),
            predicted_price=round(predicted_price, 2),
            change_percent=round(change_percent, 2),
            direction=direction,
            confidence=confidence,
            model_type=model_type,
            indicators=indicators,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Previsão concluída: {resolved_symbol} ${predicted_price:.2f} ({change_percent:+.2f}%)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na previsão para {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro na previsão: {str(e)}"
        )


@router.get("/history/recent")
async def get_history(
    request: Request,
    symbol: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Histórico de previsões realizadas.
    
    Retorna previsões salvas no PostgreSQL.
    """
    db = getattr(request.app.state, 'db', None)
    
    if not db:
        return {
            "message": "Database não configurado",
            "history": [],
            "total": 0
        }
    
    try:
        history = db.get_predictions_history(symbol=symbol, limit=limit)
        
        return {
            "history": history,
            "total": len(history),
            "filter": symbol
        }
    except Exception as e:
        logger.error(f"❌ Erro ao buscar histórico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/available")
async def get_available_models(request: Request):
    """Lista modelos disponíveis."""
    model_service: ModelService = request.app.state.model_service
    
    return {
        "models": model_service.list_available_models(),
        "hub_repo": "henriquebap/stock-predictor-lstm",
        "model_types": {
            "BASE": "Modelo genérico treinado com múltiplas ações",
            "SPECIFIC": "Modelos especializados para ações individuais"
        }
    }


@router.get("/models/info/{symbol}")
async def get_model_info(request: Request, symbol: str):
    """
    Retorna informações detalhadas sobre o modelo em uso para um símbolo.
    
    Útil para verificar:
    - Qual modelo está sendo usado (específico ou BASE)
    - Arquitetura do modelo (original ou improved)
    - Parâmetros do modelo (hidden_size, num_layers, etc.)
    """
    model_service: ModelService = request.app.state.model_service
    
    info = model_service.get_model_info(symbol.upper())
    
    return info


@router.get("/models/loaded")
async def get_loaded_models(request: Request):
    """
    Lista todos os modelos atualmente carregados em cache.
    
    Útil para monitoramento e debug.
    """
    model_service: ModelService = request.app.state.model_service
    
    loaded = model_service.get_all_loaded_models()
    
    return {
        "total_loaded": len(loaded),
        "models": loaded
    }


@router.get("/performance")
async def get_model_performance(request: Request, symbol: Optional[str] = None):
    """
    Performance histórica dos modelos.
    
    Retorna métricas de acurácia das previsões passadas.
    """
    db = getattr(request.app.state, 'db', None)
    
    if not db:
        return {
            "message": "Database não configurado",
            "performance": []
        }
    
    try:
        performance = db.get_model_performance(symbol=symbol)
        
        return {
            "performance": performance,
            "symbol": symbol
        }
    except Exception as e:
        logger.error(f"❌ Erro ao buscar performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
