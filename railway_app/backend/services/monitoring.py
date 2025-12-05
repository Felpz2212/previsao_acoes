"""
Monitoring Service - Rastreia performance do modelo e da API em produção.

Métricas coletadas:
- Tempo de resposta por endpoint
- Tempo de inferência do modelo
- Utilização de CPU e memória
- Contagem de requisições
- Taxa de erros
"""
import time
import psutil
import threading
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class RequestMetric:
    """Métrica de uma requisição individual."""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    timestamp: datetime
    model_inference_time_ms: Optional[float] = None
    symbol: Optional[str] = None


@dataclass
class SystemMetrics:
    """Métricas de sistema."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    timestamp: datetime


class MonitoringService:
    """Serviço de monitoramento de performance."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.start_time = datetime.utcnow()
        
        # Histórico de requisições
        self.requests: List[RequestMetric] = []
        
        # Contadores
        self.request_count = 0
        self.error_count = 0
        self.prediction_count = 0
        
        # Métricas agregadas por endpoint
        self.endpoint_stats: Dict[str, dict] = defaultdict(lambda: {
            'count': 0,
            'total_time_ms': 0,
            'min_time_ms': float('inf'),
            'max_time_ms': 0,
            'errors': 0
        })
        
        # Métricas de modelo
        self.model_stats: Dict[str, dict] = defaultdict(lambda: {
            'count': 0,
            'total_time_ms': 0,
            'min_time_ms': float('inf'),
            'max_time_ms': 0
        })
        
        # Métricas de sistema (últimas 100)
        self.system_metrics: List[SystemMetrics] = []
        
        # Lock para thread safety
        self._lock = threading.Lock()
        
        logger.info("📊 MonitoringService inicializado")
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        model_inference_time_ms: Optional[float] = None,
        symbol: Optional[str] = None
    ):
        """Registra métricas de uma requisição."""
        with self._lock:
            # Criar métrica
            metric = RequestMetric(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                timestamp=datetime.utcnow(),
                model_inference_time_ms=model_inference_time_ms,
                symbol=symbol
            )
            
            # Adicionar ao histórico
            self.requests.append(metric)
            if len(self.requests) > self.max_history:
                self.requests.pop(0)
            
            # Atualizar contadores
            self.request_count += 1
            if status_code >= 400:
                self.error_count += 1
            
            # Atualizar stats do endpoint
            stats = self.endpoint_stats[endpoint]
            stats['count'] += 1
            stats['total_time_ms'] += response_time_ms
            stats['min_time_ms'] = min(stats['min_time_ms'], response_time_ms)
            stats['max_time_ms'] = max(stats['max_time_ms'], response_time_ms)
            if status_code >= 400:
                stats['errors'] += 1
            
            # Se teve inferência de modelo
            if model_inference_time_ms is not None:
                self.prediction_count += 1
                model_key = symbol or 'unknown'
                model_stats = self.model_stats[model_key]
                model_stats['count'] += 1
                model_stats['total_time_ms'] += model_inference_time_ms
                model_stats['min_time_ms'] = min(model_stats['min_time_ms'], model_inference_time_ms)
                model_stats['max_time_ms'] = max(model_stats['max_time_ms'], model_inference_time_ms)
    
    def record_system_metrics(self):
        """Coleta métricas de sistema."""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metric = SystemMetrics(
                cpu_percent=cpu,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                timestamp=datetime.utcnow()
            )
            
            with self._lock:
                self.system_metrics.append(metric)
                if len(self.system_metrics) > 100:
                    self.system_metrics.pop(0)
            
            return metric
        except Exception as e:
            logger.warning(f"Erro ao coletar métricas de sistema: {e}")
            return None
    
    def get_summary(self) -> dict:
        """Retorna resumo de todas as métricas."""
        with self._lock:
            uptime = datetime.utcnow() - self.start_time
            
            # Calcular médias
            endpoint_summary = {}
            for endpoint, stats in self.endpoint_stats.items():
                if stats['count'] > 0:
                    endpoint_summary[endpoint] = {
                        'count': stats['count'],
                        'avg_time_ms': round(stats['total_time_ms'] / stats['count'], 2),
                        'min_time_ms': round(stats['min_time_ms'], 2) if stats['min_time_ms'] != float('inf') else 0,
                        'max_time_ms': round(stats['max_time_ms'], 2),
                        'error_rate': round(stats['errors'] / stats['count'] * 100, 2)
                    }
            
            # Model stats
            model_summary = {}
            for symbol, stats in self.model_stats.items():
                if stats['count'] > 0:
                    model_summary[symbol] = {
                        'predictions': stats['count'],
                        'avg_inference_ms': round(stats['total_time_ms'] / stats['count'], 2),
                        'min_inference_ms': round(stats['min_time_ms'], 2) if stats['min_time_ms'] != float('inf') else 0,
                        'max_inference_ms': round(stats['max_time_ms'], 2)
                    }
            
            # System metrics (última)
            system = None
            if self.system_metrics:
                latest = self.system_metrics[-1]
                system = {
                    'cpu_percent': latest.cpu_percent,
                    'memory_percent': latest.memory_percent,
                    'memory_used_mb': round(latest.memory_used_mb, 2),
                    'memory_available_mb': round(latest.memory_available_mb, 2)
                }
            
            return {
                'uptime_seconds': int(uptime.total_seconds()),
                'uptime_human': str(uptime).split('.')[0],
                'total_requests': self.request_count,
                'total_errors': self.error_count,
                'error_rate_percent': round(self.error_count / max(self.request_count, 1) * 100, 2),
                'total_predictions': self.prediction_count,
                'requests_per_minute': round(self.request_count / max(uptime.total_seconds() / 60, 1), 2),
                'endpoints': endpoint_summary,
                'models': model_summary,
                'system': system,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_recent_requests(self, limit: int = 50) -> List[dict]:
        """Retorna requisições recentes."""
        with self._lock:
            recent = self.requests[-limit:]
            return [
                {
                    'endpoint': r.endpoint,
                    'method': r.method,
                    'status_code': r.status_code,
                    'response_time_ms': round(r.response_time_ms, 2),
                    'model_inference_ms': round(r.model_inference_time_ms, 2) if r.model_inference_time_ms else None,
                    'symbol': r.symbol,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in reversed(recent)
            ]
    
    def get_system_history(self) -> List[dict]:
        """Retorna histórico de métricas de sistema."""
        with self._lock:
            return [
                {
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'memory_used_mb': round(m.memory_used_mb, 2),
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.system_metrics
            ]


# Instância global
_monitoring: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Obtém instância do MonitoringService (singleton)."""
    global _monitoring
    if _monitoring is None:
        _monitoring = MonitoringService()
    return _monitoring

