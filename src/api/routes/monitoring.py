"""
Monitoring and metrics endpoints.
"""
from fastapi import APIRouter
from fastapi.responses import Response
from loguru import logger

from src.api.schemas import MetricsResponse
from src.utils.monitoring import metrics_collector


router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get API metrics.
    
    Returns:
        Metrics response with API statistics
    """
    try:
        metrics = metrics_collector.get_metrics()
        return MetricsResponse(**metrics)
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        return MetricsResponse(
            total_requests=0,
            total_predictions=0,
            average_response_time=0,
            uptime_seconds=0,
            timestamp=metrics['timestamp']
        )


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format.
    
    Returns:
        Prometheus-formatted metrics
    """
    try:
        metrics_data = metrics_collector.get_prometheus_metrics()
        return Response(content=metrics_data, media_type="text/plain")
    except Exception as e:
        logger.error(f"Error fetching Prometheus metrics: {str(e)}")
        return Response(content="", media_type="text/plain")

