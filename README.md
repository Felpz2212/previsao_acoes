# Stock Price Prediction API - LSTM Neural Network

[![CI/CD](https://github.com/your-username/previsao_acoes/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/your-username/previsao_acoes/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um sistema completo de previsão de preços de ações usando redes neurais LSTM (Long Short-Term Memory), com API RESTful construída em FastAPI, monitoramento em tempo real e deploy automatizado.

> **📚 Documentação Completa Disponível!**
> 
> Acesse a pasta [`docs/`](docs/) para documentação detalhada com 300+ páginas e 35+ diagramas:
> 
> | Documento | Descrição |
> |-----------|-----------|
> | ⭐ [README_COMPLETO.md](docs/README_COMPLETO.md) | Visão geral completa (~80 páginas) |
> | 📊 [GUIA_VISUAL.md](docs/GUIA_VISUAL.md) | 35+ diagramas de fluxos |
> | 🏗️ [ARQUITETURA_TECNICA.md](docs/ARQUITETURA_TECNICA.md) | Detalhes técnicos |
> | 📋 [REFERENCIA_RAPIDA.md](docs/REFERENCIA_RAPIDA.md) | Cheat sheet de comandos |
> | ⚡ [QUICKSTART.md](docs/QUICKSTART.md) | Setup em 5 minutos |
> | 🎤 [APRESENTACAO.md](docs/APRESENTACAO.md) | Roteiro de apresentação |
> | 🌐 [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Guias de deploy |
> 
> 👉 **Comece por:** [docs/LEIA_ME_PRIMEIRO.md](docs/LEIA_ME_PRIMEIRO.md)

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura](#arquitetura)
- [Funcionalidades](#funcionalidades)
- [Instalação](#instalação)
- [Uso](#uso)
- [API Endpoints](#api-endpoints)
- [Treinamento de Modelos](#treinamento-de-modelos)
- [Deploy](#deploy)
- [Testes](#testes)
- [Monitoramento](#monitoramento)
- [Contribuindo](#contribuindo)

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido como parte do **Tech Challenge Fase 4** da Pós-Tech FIAP em Machine Learning Engineering. O objetivo é criar um sistema de ponta a ponta para previsão de preços de ações utilizando:

- **Deep Learning**: Modelo LSTM para capturar padrões temporais
- **Feature Engineering**: Indicadores técnicos, médias móveis, volatilidade
- **API RESTful**: FastAPI com endpoints para previsões, dados históricos e gerenciamento de modelos
- **Containerização**: Docker e Docker Compose
- **CI/CD**: GitHub Actions para testes e deploy automatizados
- **Monitoramento**: Métricas Prometheus e logging estruturado

### Métricas de Avaliação

O modelo é avaliado usando:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **Directional Accuracy** (acurácia da direção da mudança de preço)

## 🏗️ Arquitetura

```
┌─────────────────┐
│   Yahoo Finance │
│   (Data Source) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│     Data Collection & Processing        │
│  • Data Loader (yfinance)               │
│  • Feature Engineering                  │
│  • Data Preprocessing                   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        LSTM Model Training              │
│  • PyTorch LSTM                         │
│  • Sequence Generation                  │
│  • Model Evaluation (MAE, RMSE, MAPE)  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        FastAPI REST API                 │
│  • Prediction Endpoints                 │
│  • Model Management                     │
│  • Historical Data Access               │
│  • Monitoring & Metrics                 │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│        Deployment Options               │
│  • Railway (Backend + API)              │
│  • HuggingFace Spaces (UI/Demo)         │
│  • Docker Containers                    │
└─────────────────────────────────────────┘
```

## ✨ Funcionalidades

### Core Features

- ✅ **Previsão de Preços**: Previsão de preços de fechamento de ações usando LSTM
- ✅ **Múltiplas Ações**: Suporte para previsão de múltiplas ações
- ✅ **Previsões em Lote**: API endpoint para previsões batch
- ✅ **Dados Históricos**: Acesso a dados históricos via API
- ✅ **Treinamento Automático**: Pipeline completo de treinamento com validação
- ✅ **Retreinamento**: Endpoint para retreinar modelos sob demanda

### API & Monitoring

- ✅ **API RESTful**: FastAPI com documentação automática (Swagger/OpenAPI)
- ✅ **Monitoramento**: Métricas Prometheus e dashboard de monitoramento
- ✅ **Logging Estruturado**: Logs detalhados com Loguru
- ✅ **Health Checks**: Endpoints de saúde da aplicação
- ✅ **CORS**: Configuração CORS para integração frontend

### DevOps

- ✅ **Containerização**: Docker e Docker Compose
- ✅ **CI/CD**: GitHub Actions
- ✅ **Testes**: Suite de testes com pytest
- ✅ **Agendamento**: Scripts para treinamento agendado (cron)

## 🚀 Instalação

### Pré-requisitos

- Python 3.10+
- Docker (opcional, para containerização)
- Git

### Instalação Local

1. **Clone o repositório**

```bash
git clone https://github.com/your-username/previsao_acoes.git
cd previsao_acoes
```

2. **Crie um ambiente virtual**

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. **Instale as dependências**

```bash
pip install -r requirements.txt
```

4. **Configure variáveis de ambiente**

```bash
cp .env.example .env
# Edite .env com suas configurações
```

### Instalação com Docker

```bash
docker-compose up --build
```

A API estará disponível em `http://localhost:8000`

## 📖 Uso

### 1. Treinar um Modelo

```bash
# Treinar modelo para Apple (AAPL)
python scripts/train_model.py AAPL --start-date 2018-01-01 --end-date 2024-12-31

# Com opções personalizadas
python scripts/train_model.py GOOGL --epochs 100 --batch-size 64
```

### 2. Iniciar a API

```bash
# Desenvolvimento
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Produção
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Acessar a Documentação Interativa

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 4. Fazer Previsões via API

```bash
# Previsão simples
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "days_ahead": 1
  }'

# Previsão em lote
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "days_ahead": 1
  }'
```

## 🔌 API Endpoints

### Previsões

- `POST /api/v1/predict` - Previsão para uma ação
- `POST /api/v1/predict/batch` - Previsões em lote

### Dados

- `GET /api/v1/stocks/{symbol}/historical` - Dados históricos
- `GET /api/v1/stocks/{symbol}/latest` - Preço mais recente
- `GET /api/v1/stocks/available` - Lista de ações disponíveis

### Modelos

- `POST /api/v1/models/train` - Treinar/retreinar modelo
- `GET /api/v1/models/status` - Status de todos os modelos
- `GET /api/v1/models/{symbol}/performance` - Métricas de um modelo

### Monitoramento

- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - Métricas da API
- `GET /api/v1/metrics/prometheus` - Métricas em formato Prometheus

## 🎓 Treinamento de Modelos

### Pipeline de Treinamento

O processo de treinamento inclui:

1. **Coleta de Dados**: Download de dados históricos do Yahoo Finance
2. **Feature Engineering**: Criação de features técnicas
   - Médias móveis (7, 30, 90 dias)
   - Volatilidade
   - Indicadores de momentum
   - Features baseadas em volume
3. **Preprocessamento**: Normalização e criação de sequências
4. **Treinamento**: LSTM com validação
5. **Avaliação**: Cálculo de métricas (RMSE, MAE, MAPE, R²)
6. **Salvamento**: Modelo e preprocessador salvos para inferência

### Configuração do Modelo

Edite `config/settings.py` para ajustar hiperparâmetros:

```python
LSTM_SEQUENCE_LENGTH = 60  # Dias de histórico
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001
LSTM_HIDDEN_SIZE = 50
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
```

## 🚢 Deploy

### Railway

1. **Conecte seu repositório ao Railway**
2. **Configure variáveis de ambiente**
3. **Deploy automático via push no main**

Veja [DEPLOYMENT.md](docs/DEPLOYMENT.md) para instruções detalhadas.

### HuggingFace Spaces (UI Demo)

Crie um Gradio app em `app_gradio.py` e faça deploy no HuggingFace Spaces.

### Docker

```bash
# Build
docker build -t stock-prediction-api .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  stock-prediction-api
```

## 🧪 Testes

```bash
# Todos os testes
pytest

# Com cobertura
pytest --cov=src tests/

# Testes específicos
pytest tests/test_api.py -v

# Ignorar testes lentos
pytest -m "not slow"
```

## 📊 Monitoramento

### Métricas Disponíveis

- **API Metrics**: Requests totais, latência média, uptime
- **Model Metrics**: Tempo de predição, acurácia, erro
- **System Metrics**: CPU, memória, disco

### Prometheus Integration

Métricas disponíveis em formato Prometheus:

```
http://localhost:8000/api/v1/metrics/prometheus
```

### Logging

Logs estruturados são salvos em:
- `logs/app_YYYY-MM-DD.log` - Todos os logs
- `logs/errors_YYYY-MM-DD.log` - Apenas erros

## 📁 Estrutura do Projeto

```
previsao_acoes/
├── src/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # Main app
│   │   ├── schemas.py    # Pydantic models
│   │   └── routes/       # API routes
│   ├── data/             # Data handling
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/           # ML models
│   │   └── lstm_model.py
│   ├── training/         # Training pipeline
│   │   └── trainer.py
│   └── utils/            # Utilities
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── config/               # Configuration
├── models/               # Saved models
├── data/                 # Data storage
├── logs/                 # Application logs
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose
├── requirements.txt      # Python dependencies
└── README.md
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👨‍💻 Autor

**Seu Nome**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-profile](https://linkedin.com/in/your-profile)

## 🙏 Agradecimentos

- FIAP Pós-Tech MLET
- Tech Challenge Fase 4
- Comunidade Python/PyTorch
- Colaboradores e revisores

## 📚 Referências

- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Yahoo Finance API](https://github.com/ranaroussi/yfinance)

---

**Nota**: Este é um projeto educacional. Não use para decisões reais de investimento sem análise adicional e consultoria profissional.

