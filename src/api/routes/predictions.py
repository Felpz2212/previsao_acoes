"""
Prediction endpoints for stock price forecasting.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime, timedelta
import time
from pathlib import Path
from loguru import logger

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from src.models.lstm_model import LSTMPredictor
from src.data.preprocessor import StockDataPreprocessor
from src.data.data_loader import StockDataLoader
from src.utils.monitoring import metrics_collector
from config.settings import get_model_path, get_scaler_path, DEFAULT_START_DATE


router = APIRouter()


def load_model_and_preprocessor(symbol: str):
    """
    Load trained model and preprocessor for a stock symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Tuple of (model, preprocessor)
    """
    model_path = get_model_path(symbol)
    scaler_path = get_scaler_path(symbol)
    
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for symbol {symbol}. Please train the model first."
        )
    
    if not scaler_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Preprocessor not found for symbol {symbol}. Please train the model first."
        )
    
    try:
        model = LSTMPredictor.load(model_path)
        preprocessor = StockDataPreprocessor.load(scaler_path)
        return model, preprocessor
    except Exception as e:
        logger.error(f"Error loading model for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """
    Predict future stock price for a given symbol.
    
    Args:
        request: Prediction request with symbol and days ahead
        
    Returns:
        Prediction response with predicted price
    """
    start_time = time.time()
    
    try:
        logger.info(f"Prediction request for {request.symbol}")
        
        # Load model and preprocessor
        model, preprocessor = load_model_and_preprocessor(request.symbol)
        
        # Load recent historical data
        data_loader = StockDataLoader()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        df = data_loader.load_stock_data(
            request.symbol,
            start_date,
            end_date
        )
        
        if len(df) < preprocessor.sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data. Need at least {preprocessor.sequence_length} days."
            )
        
        # Get current price
        current_price = float(df['close'].iloc[-1])
        
        # Prepare data for prediction
        X_pred = preprocessor.transform_for_prediction(df)
        
        # Make prediction
        prediction_scaled = model.predict(X_pred)[0]
        
        # Note: For proper inverse transform, we need to handle this carefully
        # For now, we'll use a simple approach
        predicted_price = float(prediction_scaled)
        
        # Record metrics
        duration = time.time() - start_time
        metrics_collector.record_prediction(request.symbol, duration)
        
        # Calculate prediction date
        prediction_date = (datetime.now() + timedelta(days=request.days_ahead)).strftime("%Y-%m-%d")
        
        response = PredictionResponse(
            symbol=request.symbol,
            current_price=current_price,
            predicted_price=predicted_price,
            prediction_date=prediction_date,
            timestamp=datetime.now()
        )
        
        logger.info(
            f"Prediction for {request.symbol}: "
            f"current={current_price:.2f}, predicted={predicted_price:.2f}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict_stock_prices(request: BatchPredictionRequest):
    """
    Predict future stock prices for multiple symbols.
    
    Args:
        request: Batch prediction request with list of symbols
        
    Returns:
        Batch prediction response with predictions for all symbols
    """
    predictions = []
    
    for symbol in request.symbols:
        try:
            pred_request = PredictionRequest(
                symbol=symbol,
                days_ahead=request.days_ahead
            )
            prediction = await predict_stock_price(pred_request)
            predictions.append(prediction)
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {str(e)}")
            # Continue with other symbols even if one fails
            continue
    
    return BatchPredictionResponse(
        predictions=predictions,
        timestamp=datetime.now()
    )

