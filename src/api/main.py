"""
Main FastAPI application for stock price prediction.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from datetime import datetime
import time
from loguru import logger

from src.utils.logger import setup_logging
from src.utils.monitoring import metrics_collector, ACTIVE_REQUESTS
from src.api.routes import predictions, data, models, monitoring
from src.api.schemas import ErrorResponse


# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing and metrics middleware
@app.middleware("http")
async def add_process_time_and_metrics(request: Request, call_next):
    """Add request processing time and collect metrics."""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Record metrics
        metrics_collector.record_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=process_time
        )
        
        logger.info(
            f"{request.method} {request.url.path} "
            f"completed in {process_time:.3f}s "
            f"with status {response.status_code}"
        )
        
        return response
    
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {str(e)}")
        
        # Record failed request
        metrics_collector.record_request(
            method=request.method,
            endpoint=request.url.path,
            status=500,
            duration=process_time
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                detail=str(e),
                timestamp=datetime.now().isoformat()
            ).model_dump()
        )
    
    finally:
        ACTIVE_REQUESTS.dec()


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


# Include routers
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(data.router, prefix="/api/v1", tags=["Data"])
app.include_router(models.router, prefix="/api/v1", tags=["Models"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Stock Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT, API_WORKERS
    
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        reload=True,  # Set to False in production
        log_level="info"
    )

