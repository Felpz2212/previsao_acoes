"""
Monitoring and metrics collection utilities.
"""
import time
from typing import Dict
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from loguru import logger


# Prometheus metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['symbol']
)

MODEL_PREDICTION_TIME = Histogram(
    'model_prediction_seconds',
    'Time taken for model prediction',
    ['symbol']
)

ACTIVE_REQUESTS = Gauge(
    'api_active_requests',
    'Number of active requests'
)


class MetricsCollector:
    """Collect and manage application metrics."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.start_time = time.time()
        self.total_requests = 0
        self.total_predictions = 0
        self.response_times = []
        self.max_response_times = 1000  # Keep last N response times
        
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """
        Record an API request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status: HTTP status code
            duration: Request duration in seconds
        """
        self.total_requests += 1
        self.response_times.append(duration)
        
        # Keep only last N response times
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)
        
        # Update Prometheus metrics
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_prediction(self, symbol: str, duration: float):
        """
        Record a prediction.
        
        Args:
            symbol: Stock symbol
            duration: Prediction duration in seconds
        """
        self.total_predictions += 1
        PREDICTION_COUNT.labels(symbol=symbol).inc()
        MODEL_PREDICTION_TIME.labels(symbol=symbol).observe(duration)
    
    def get_metrics(self) -> Dict:
        """
        Get current metrics.
        
        Returns:
            Dictionary with metrics
        """
        uptime = time.time() - self.start_time
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )
        
        return {
            'total_requests': self.total_requests,
            'total_predictions': self.total_predictions,
            'average_response_time': avg_response_time,
            'uptime_seconds': uptime,
            'timestamp': datetime.now()
        }
    
    def get_prometheus_metrics(self) -> bytes:
        """
        Get metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus format
        """
        return generate_latest()


# Global metrics collector instance
metrics_collector = MetricsCollector()

