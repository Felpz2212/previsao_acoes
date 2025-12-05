"""
Model management endpoints for training and status.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from loguru import logger
import json
from pathlib import Path

from src.api.schemas import (
    TrainingRequest,
    TrainingResponse,
    ModelStatusResponse,
    ModelInfo
)
from src.training.trainer import train_stock_model
from config.settings import MODELS_DIR


router = APIRouter()


# Background task for training
def train_model_background(symbol: str, start_date: str, end_date: str):
    """
    Background task to train a model.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date for training data
        end_date: End date for training data
    """
    try:
        logger.info(f"Starting background training for {symbol}")
        metrics = train_stock_model(symbol, start_date, end_date)
        logger.info(f"Training completed for {symbol}: {metrics}")
    except Exception as e:
        logger.error(f"Error in background training for {symbol}: {str(e)}")


@router.post("/models/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train or retrain a model for a stock symbol.
    
    Args:
        request: Training request
        background_tasks: FastAPI background tasks
        
    Returns:
        Training response
    """
    try:
        logger.info(f"Training request for {request.symbol}")
        
        # Check if model already exists
        model_path = MODELS_DIR / f"lstm_model_{request.symbol}.pth"
        
        if model_path.exists() and not request.force_retrain:
            return TrainingResponse(
                symbol=request.symbol,
                status="exists",
                message="Model already exists. Use force_retrain=true to retrain.",
                trained_at=None
            )
        
        # Add training task to background
        background_tasks.add_task(
            train_model_background,
            request.symbol,
            request.start_date,
            request.end_date
        )
        
        return TrainingResponse(
            symbol=request.symbol,
            status="training",
            message="Training started in background. Check status endpoint for progress.",
            trained_at=None
        )
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting training: {str(e)}"
        )


@router.get("/models/status", response_model=ModelStatusResponse)
async def get_models_status():
    """
    Get status of all trained models.
    
    Returns:
        Model status response with list of all models
    """
    try:
        logger.info("Fetching models status")
        
        # Find all metadata files
        metadata_files = list(MODELS_DIR.glob("metadata_*.json"))
        
        models = []
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                    models.append(
                        ModelInfo(
                            symbol=metadata['symbol'],
                            trained_at=datetime.fromisoformat(metadata['trained_at']),
                            metrics=metadata.get('metrics', {}),
                            model_path=metadata['model_path'],
                            sequence_length=metadata['sequence_length'],
                            epochs=metadata['epochs']
                        )
                    )
            except Exception as e:
                logger.error(f"Error reading metadata file {metadata_file}: {str(e)}")
                continue
        
        return ModelStatusResponse(
            models=models,
            count=len(models)
        )
        
    except Exception as e:
        logger.error(f"Error fetching models status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching models status: {str(e)}"
        )


@router.get("/models/{symbol}/performance")
async def get_model_performance(symbol: str):
    """
    Get detailed performance metrics for a specific model.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Model performance metrics
    """
    try:
        metadata_file = MODELS_DIR / f"metadata_{symbol}.json"
        
        if not metadata_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for symbol {symbol}"
            )
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return {
            'symbol': metadata['symbol'],
            'trained_at': metadata['trained_at'],
            'metrics': metadata.get('metrics', {}),
            'hyperparameters': {
                'sequence_length': metadata['sequence_length'],
                'epochs': metadata['epochs'],
                'batch_size': metadata['batch_size'],
                'learning_rate': metadata['learning_rate'],
                'hidden_size': metadata['hidden_size'],
                'num_layers': metadata['num_layers'],
                'dropout': metadata['dropout']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching model performance: {str(e)}"
        )

