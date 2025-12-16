"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"


class PredictionRequest(BaseModel):
    """Request for stock price prediction."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")
    days_ahead: int = Field(1, ge=1, le=30, description="Number of days to predict ahead")


class PredictionResponse(BaseModel):
    """Response with stock price prediction."""
    symbol: str
    current_price: float
    predicted_price: float
    prediction_date: str
    confidence: Optional[float] = None
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """Request for multiple stock predictions."""
    symbols: List[str] = Field(..., description="List of stock ticker symbols")
    days_ahead: int = Field(1, ge=1, le=30, description="Number of days to predict ahead")


class BatchPredictionResponse(BaseModel):
    """Response with multiple stock predictions."""
    predictions: List[PredictionResponse]
    timestamp: datetime


class HistoricalDataRequest(BaseModel):
    """Request for historical stock data."""
    symbol: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: Optional[int] = Field(100, ge=1, le=1000)


class StockDataPoint(BaseModel):
    """Single stock data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class HistoricalDataResponse(BaseModel):
    """Response with historical stock data."""
    symbol: str
    data: List[StockDataPoint]
    count: int


class TrainingRequest(BaseModel):
    """Request to train/retrain a model."""
    symbol: str
    start_date: str
    end_date: str
    force_retrain: bool = Field(False, description="Force retraining even if model exists")


class TrainingResponse(BaseModel):
    """Response for training request."""
    symbol: str
    status: str
    message: str
    metrics: Optional[Dict[str, float]] = None
    trained_at: Optional[datetime] = None


class ModelInfo(BaseModel):
    """Information about a trained model."""
    symbol: str
    trained_at: datetime
    metrics: Dict[str, float]
    model_path: str
    sequence_length: int
    epochs: int


class ModelStatusResponse(BaseModel):
    """Response with model status information."""
    models: List[ModelInfo]
    count: int


class MetricsResponse(BaseModel):
    """Response with API metrics."""
    total_requests: int
    total_predictions: int
    average_response_time: float
    uptime_seconds: float
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: str  # ISO format string for JSON serialization

