# Monitoramento - Prometheus + Grafana

Sistema de monitoramento para a API Stock Predictor.

## Requisitos

- Docker e Docker Compose
- API rodando (local ou Railway)

## Iniciar Monitoramento

```bash
cd monitoring
docker-compose up -d
```

## Acessos

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin / admin |

## Métricas Coletadas

### Requisições HTTP
- `stock_predictor_requests_total` - Total de requisições por método/endpoint/status
- `stock_predictor_request_duration_seconds` - Histograma de latência

### Previsões
- `stock_predictor_predictions_total` - Total de previsões por símbolo
- `stock_predictor_model_inference_seconds` - Tempo de inferência

### Qualidade do Modelo
- `stock_predictor_model_mape` - MAPE por símbolo
- `stock_predictor_model_directional_accuracy` - Acurácia direcional

### Sistema
- `stock_predictor_system_cpu_percent` - Uso de CPU
- `stock_predictor_system_memory_percent` - Uso de memória

## Dashboard Grafana

O dashboard pré-configurado inclui:

1. **Overview**
   - Status da API
   - Requests/segundo
   - Total de previsões
   - CPU e Memória

2. **Request Latency**
   - Percentis p50, p95, p99
   - Requests por status code

3. **Model Performance**
   - Tempo de inferência por símbolo
   - MAPE por modelo

4. **System Resources**
   - CPU e Memória ao longo do tempo

## Configuração para Railway

Para monitorar a API em produção, o `prometheus.yml` já está configurado:

```yaml
- job_name: 'stock-predictor-railway'
  static_configs:
    - targets: ['previsaoacoes-back-production.up.railway.app']
  scheme: https
  metrics_path: '/metrics'
```

## Endpoints da API

| Endpoint | Descrição |
|----------|-----------|
| `/metrics` | Métricas Prometheus (formato padrão) |
| `/api/monitoring` | Resumo JSON de métricas |
| `/api/monitoring/requests` | Requisições recentes |
| `/api/monitoring/system` | Histórico de sistema |

## Parar Monitoramento

```bash
docker-compose down
```

Para remover dados:
```bash
docker-compose down -v
```



