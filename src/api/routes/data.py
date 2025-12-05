"""
Data endpoints for historical stock data.
"""
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from src.api.schemas import (
    HistoricalDataResponse,
    StockDataPoint
)
from src.data.data_loader import StockDataLoader


router = APIRouter()


@router.get("/stocks/{symbol}/historical", response_model=HistoricalDataResponse)
async def get_historical_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records")
):
    """
    Get historical stock data for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date (optional)
        end_date: End date (optional)
        limit: Maximum number of records to return
        
    Returns:
        Historical data response
    """
    try:
        logger.info(f"Fetching historical data for {symbol}")
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if start_date is None:
            # Default to 1 year ago
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Load data
        data_loader = StockDataLoader()
        df = data_loader.load_stock_data(symbol, start_date, end_date)
        
        # Limit the number of records
        if len(df) > limit:
            df = df.tail(limit)
        
        # Convert to response format
        data_points = []
        for _, row in df.iterrows():
            data_points.append(
                StockDataPoint(
                    timestamp=row['timestamp'],
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )
            )
        
        return HistoricalDataResponse(
            symbol=symbol,
            data=data_points,
            count=len(data_points)
        )
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching historical data: {str(e)}"
        )


@router.get("/stocks/{symbol}/latest")
async def get_latest_price(symbol: str):
    """
    Get the latest price information for a stock.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Latest price information
    """
    try:
        logger.info(f"Fetching latest price for {symbol}")
        
        data_loader = StockDataLoader()
        latest_info = data_loader.get_latest_price(symbol)
        
        return latest_info
        
    except Exception as e:
        logger.error(f"Error fetching latest price: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching latest price: {str(e)}"
        )


@router.get("/stocks/available")
async def get_available_stocks():
    """
    Get list of available stocks with trained models.
    
    Returns:
        List of available stock symbols
    """
    try:
        from config.settings import MODELS_DIR
        import json
        
        # Find all metadata files
        metadata_files = list(MODELS_DIR.glob("metadata_*.json"))
        
        available_stocks = []
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                available_stocks.append({
                    'symbol': metadata['symbol'],
                    'trained_at': metadata['trained_at'],
                    'metrics': metadata.get('metrics', {})
                })
        
        return {
            'stocks': available_stocks,
            'count': len(available_stocks)
        }
        
    except Exception as e:
        logger.error(f"Error fetching available stocks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching available stocks: {str(e)}"
        )

