"""
Stocks Routes - Dados históricos de ações
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from services.stock_service import StockService


router = APIRouter()
stock_service = StockService()


# Schemas
class StockData(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockResponse(BaseModel):
    symbol: str
    name: str
    current_price: float
    change_percent: float
    data: List[dict]
    indicators: dict


class CompareResponse(BaseModel):
    symbols: List[str]
    data: dict


# Mapeamento de empresas
COMPANY_MAP = {
    "apple": "AAPL", "maçã": "AAPL",
    "google": "GOOGL", "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "meta": "META", "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "petrobras": "PETR4.SA",
    "vale": "VALE3.SA",
    "itau": "ITUB4.SA", "itaú": "ITUB4.SA",
    "bradesco": "BBDC4.SA",
    "ambev": "ABEV3.SA",
    "nubank": "NU",
    "mercado livre": "MELI",
}


def resolve_symbol(query: str) -> str:
    """Converte nome de empresa para ticker."""
    query = query.strip()
    if query.upper() == query and 1 <= len(query) <= 10:
        return query.upper()
    
    key = query.lower()
    if key in COMPANY_MAP:
        return COMPANY_MAP[key]
    
    for company, ticker in COMPANY_MAP.items():
        if key in company:
            return ticker
    
    return query.upper()


@router.get("/{symbol}", response_model=StockResponse)
async def get_stock(
    symbol: str,
    days: int = Query(default=365, ge=30, le=1825)
):
    """
    Obtém dados históricos de uma ação.
    
    - **symbol**: Ticker ou nome da empresa (ex: AAPL, Apple, Petrobras)
    - **days**: Número de dias de histórico (30-1825)
    """
    resolved_symbol = resolve_symbol(symbol)
    
    try:
        data = stock_service.get_stock_data(resolved_symbol, days)
        
        if data is None or len(data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Dados não encontrados para {resolved_symbol}"
            )
        
        # Calcular indicadores
        current_price = float(data['close'].iloc[-1])
        prev_price = float(data['close'].iloc[-2]) if len(data) > 1 else current_price
        change_percent = ((current_price - prev_price) / prev_price) * 100
        
        # Médias móveis
        ma_7 = float(data['close'].rolling(7).mean().iloc[-1])
        ma_30 = float(data['close'].rolling(30).mean().iloc[-1])
        
        # Volatilidade
        volatility = float(data['close'].rolling(14).std().iloc[-1])
        
        return StockResponse(
            symbol=resolved_symbol,
            name=stock_service.get_company_name(resolved_symbol),
            current_price=current_price,
            change_percent=round(change_percent, 2),
            data=data.to_dict('records'),
            indicators={
                "ma_7": round(ma_7, 2),
                "ma_30": round(ma_30, 2),
                "volatility": round(volatility, 2),
                "trend": "up" if ma_7 > ma_30 else "down"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao obter dados: {str(e)}"
        )


@router.get("/compare/multiple")
async def compare_stocks(
    symbols: str = Query(..., description="Símbolos separados por vírgula"),
    days: int = Query(default=90, ge=30, le=365)
):
    """
    Compara múltiplas ações.
    
    - **symbols**: Tickers separados por vírgula (ex: AAPL,GOOGL,MSFT)
    - **days**: Período de comparação
    """
    symbol_list = [resolve_symbol(s.strip()) for s in symbols.split(",")]
    
    if len(symbol_list) < 2:
        raise HTTPException(
            status_code=400,
            detail="Forneça pelo menos 2 símbolos para comparar"
        )
    
    if len(symbol_list) > 5:
        raise HTTPException(
            status_code=400,
            detail="Máximo de 5 ações para comparar"
        )
    
    try:
        result = {}
        for symbol in symbol_list:
            data = stock_service.get_stock_data(symbol, days)
            if data is not None:
                # Normalizar para comparação (base 100)
                base_price = float(data['close'].iloc[0])
                normalized = (data['close'] / base_price * 100).tolist()
                
                result[symbol] = {
                    "name": stock_service.get_company_name(symbol),
                    "current_price": float(data['close'].iloc[-1]),
                    "normalized": normalized,
                    "dates": data['timestamp'].astype(str).tolist(),
                    "performance": round((float(data['close'].iloc[-1]) / base_price - 1) * 100, 2)
                }
        
        return CompareResponse(
            symbols=list(result.keys()),
            data=result
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na comparação: {str(e)}"
        )


@router.get("/search/{query}")
async def search_stocks(query: str):
    """
    Busca ações por nome ou ticker.
    """
    query_lower = query.lower()
    results = []
    
    for company, ticker in COMPANY_MAP.items():
        if query_lower in company or query_lower in ticker.lower():
            results.append({
                "symbol": ticker,
                "name": company.title()
            })
    
    return {"results": results[:10]}


@router.get("/popular/list")
async def get_popular_stocks():
    """
    Lista de ações populares organizadas por categoria.
    """
    return {
        "categories": {
            "🇺🇸 Tech US": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "NFLX"],
            "💰 Finance US": ["JPM", "BAC", "V", "MA", "GS", "BRK-B"],
            "🛒 Consumer US": ["WMT", "KO", "MCD", "SBUX", "NKE", "DIS"],
            "💊 Healthcare US": ["JNJ", "PFE", "UNH", "MRNA"],
            "🇧🇷 Brasil B3": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "WEGE3.SA", "MGLU3.SA"]
        }
    }

