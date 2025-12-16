"""
Prometheus Metrics Service - Métricas em formato Prometheus.

Métricas expostas:
- stock_predictor_requests_total: Total de requisições por endpoint/método/status
- stock_predictor_request_duration_seconds: Histograma de latência
- stock_predictor_predictions_total: Total de previsões por símbolo
- stock_predictor_model_inference_seconds: Tempo de inferência do modelo
- stock_predictor_model_mape: MAPE do modelo por símbolo
- stock_predictor_system_cpu_percent: Uso de CPU
- stock_predictor_system_memory_percent: Uso de memória
"""
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess, REGISTRY
)
import psutil
import time
from typing import Optional
from loguru import logger


# Criar registry customizado para evitar conflitos
CUSTOM_REGISTRY = CollectorRegistry()

# ============== Métricas de Requisições ==============

REQUEST_COUNT = Counter(
    'stock_predictor_requests_total',
    'Total de requisições HTTP',
    ['method', 'endpoint', 'status'],
    registry=CUSTOM_REGISTRY
)

REQUEST_LATENCY = Histogram(
    'stock_predictor_request_duration_seconds',
    'Latência das requisições em segundos',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=CUSTOM_REGISTRY
)

# ============== Métricas de Previsões ==============

PREDICTION_COUNT = Counter(
    'stock_predictor_predictions_total',
    'Total de previsões realizadas',
    ['symbol', 'model_type'],
    registry=CUSTOM_REGISTRY
)

MODEL_INFERENCE_TIME = Histogram(
    'stock_predictor_model_inference_seconds',
    'Tempo de inferência do modelo em segundos',
    ['symbol'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=CUSTOM_REGISTRY
)

# ============== Métricas de Qualidade do Modelo ==============

MODEL_MAPE = Gauge(
    'stock_predictor_model_mape',
    'MAPE (Mean Absolute Percentage Error) do modelo',
    ['symbol'],
    registry=CUSTOM_REGISTRY
)

MODEL_DIRECTIONAL_ACCURACY = Gauge(
    'stock_predictor_model_directional_accuracy',
    'Acurácia direcional do modelo (0-1)',
    ['symbol'],
    registry=CUSTOM_REGISTRY
)

PREDICTION_ERROR = Gauge(
    'stock_predictor_prediction_error_percent',
    'Erro percentual da última previsão',
    ['symbol'],
    registry=CUSTOM_REGISTRY
)

# ============== Métricas de Sistema ==============

SYSTEM_CPU = Gauge(
    'stock_predictor_system_cpu_percent',
    'Porcentagem de uso de CPU',
    registry=CUSTOM_REGISTRY
)

SYSTEM_MEMORY = Gauge(
    'stock_predictor_system_memory_percent',
    'Porcentagem de uso de memória',
    registry=CUSTOM_REGISTRY
)

SYSTEM_MEMORY_USED_MB = Gauge(
    'stock_predictor_system_memory_used_mb',
    'Memória usada em MB',
    registry=CUSTOM_REGISTRY
)

# ============== Métricas de Uptime ==============

API_UP = Gauge(
    'stock_predictor_api_up',
    'API está online (1) ou offline (0)',
    registry=CUSTOM_REGISTRY
)

API_START_TIME = Gauge(
    'stock_predictor_api_start_time_seconds',
    'Timestamp de início da API',
    registry=CUSTOM_REGISTRY
)

MODELS_LOADED = Gauge(
    'stock_predictor_models_loaded',
    'Número de modelos carregados em cache',
    registry=CUSTOM_REGISTRY
)

# ============== Info ==============

API_INFO = Info(
    'stock_predictor_api',
    'Informações da API',
    registry=CUSTOM_REGISTRY
)


class PrometheusMetrics:
    """Classe para gerenciar métricas Prometheus."""
    
    def __init__(self):
        self._start_time = time.time()
        API_START_TIME.set(self._start_time)
        API_UP.set(1)
        API_INFO.info({
            'version': '1.0.0',
            'model_hub': 'henriquebap/stock-predictor-lstm',
            'framework': 'FastAPI'
        })
        logger.info("📊 PrometheusMetrics inicializado")
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration_seconds: float
    ):
        """Registra uma requisição HTTP."""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration_seconds)
    
    def record_prediction(
        self,
        symbol: str,
        model_type: str,
        inference_time_seconds: float
    ):
        """Registra uma previsão."""
        PREDICTION_COUNT.labels(
            symbol=symbol,
            model_type=model_type
        ).inc()
        
        MODEL_INFERENCE_TIME.labels(
            symbol=symbol
        ).observe(inference_time_seconds)
    
    def set_model_metrics(
        self,
        symbol: str,
        mape: Optional[float] = None,
        directional_accuracy: Optional[float] = None
    ):
        """Atualiza métricas de qualidade do modelo."""
        if mape is not None:
            MODEL_MAPE.labels(symbol=symbol).set(mape)
        if directional_accuracy is not None:
            MODEL_DIRECTIONAL_ACCURACY.labels(symbol=symbol).set(directional_accuracy)
    
    def set_prediction_error(self, symbol: str, error_percent: float):
        """Registra erro percentual de uma previsão."""
        PREDICTION_ERROR.labels(symbol=symbol).set(error_percent)
    
    def update_system_metrics(self):
        """Atualiza métricas de sistema."""
        try:
            SYSTEM_CPU.set(psutil.cpu_percent(interval=0.1))
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY.set(memory.percent)
            SYSTEM_MEMORY_USED_MB.set(memory.used / (1024 * 1024))
        except Exception as e:
            logger.warning(f"Erro ao coletar métricas de sistema: {e}")
    
    def set_models_loaded(self, count: int):
        """Atualiza contagem de modelos carregados."""
        MODELS_LOADED.set(count)
    
    def get_metrics(self) -> bytes:
        """Retorna métricas em formato Prometheus."""
        self.update_system_metrics()
        return generate_latest(CUSTOM_REGISTRY)
    
    def get_content_type(self) -> str:
        """Retorna content type para métricas Prometheus."""
        return CONTENT_TYPE_LATEST


# Instância singleton
_prometheus_metrics: Optional[PrometheusMetrics] = None


def get_prometheus_metrics() -> PrometheusMetrics:
    """Obtém instância do PrometheusMetrics (singleton)."""
    global _prometheus_metrics
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()
    return _prometheus_metrics



