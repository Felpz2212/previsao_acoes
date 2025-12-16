# Stock Price Prediction API - LSTM Neural Network

[![CI/CD](https://github.com/your-username/previsao_acoes/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/your-username/previsao_acoes/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um sistema completo de previsão de preços de ações usando redes neurais LSTM (Long Short-Term Memory), com API RESTful construída em FastAPI, monitoramento em tempo real e deploy automatizado.

## Índice

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

## Sobre o Projeto

Este projeto foi desenvolvido como parte do **Tech Challenge Fase 4** da Pós-Tech FIAP em Machine Learning Engineering. O objetivo é criar um sistema de ponta a ponta para previsão de preços de ações utilizando:

- **Deep Learning**: Modelo LSTM para capturar padrões temporais
- **Feature Engineering**: Indicadores técnicos, médias móveis, volatilidade
- **API RESTful**: FastAPI com endpoints para previsões, dados históricos e gerenciamento de modelos
- **Containerização**: Docker e Docker Compose
- **CI/CD**: GitHub Actions para testes e deploy automatizados
- **Monitoramento**: Métricas Prometheus e logging estruturado

### Tecnologias e Stack

- **Backend**: FastAPI 0.104+, Python 3.10+
- **Frontend**: Streamlit, Plotly
- **ML Framework**: PyTorch, scikit-learn
- **Database**: PostgreSQL (Railway Cloud)
- **Model Hub**: HuggingFace Hub
- **Deploy**: Railway (Docker containers)
- **Data Source**: Yahoo Finance API (yfinance)
- **Monitoring**: Prometheus metrics
- **WebSocket**: Para atualizações real-time

### Métricas de Avaliação

Os modelos são avaliados usando:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error) - Principal métrica
- **R²** (Coefficient of Determination)
- **Directional Accuracy** (acurácia da direção da mudança de preço)

## Arquitetura

### Arquitetura Completa do Sistema

```mermaid
graph TB
    subgraph "FRONTEND - Streamlit"
        UI[Interface Web<br/>Streamlit App]
        UI --> PAGE1[Página Principal<br/>- Previsões de ações<br/>- Gráficos interativos<br/>- Modo comparação]
        UI --> PAGE2[Monitoramento<br/>4 Abas especializadas]
        
        PAGE2 --> TAB1[Overview<br/>- Uptime & Requests<br/>- Taxa de erro<br/>- CPU/Memória]
        PAGE2 --> TAB2[Modelos<br/>- Métricas por modelo<br/>- Tempo de inferência<br/>- Accuracy & MAPE]
        PAGE2 --> TAB3[ML Health<br/>- Health Score 0-100<br/>- Data Drift<br/>- Prediction Analysis<br/>- Alertas automáticos]
        PAGE2 --> TAB4[Prometheus<br/>- Métricas raw<br/>- Scraping endpoint]
        
        PAGE1 --> SIDEBAR[Sidebar<br/>- Busca de ações<br/>- Ações populares<br/>- Período de análise<br/>- Modo comparação]
    end
    
    subgraph "COMUNICAÇÃO"
        HTTP[HTTP/REST<br/>Requests]
        WS[WebSocket<br/>Tempo Real]
    end
    
    subgraph "BACKEND - FastAPI"
        API[FastAPI Application<br/>Python 3.10+]
        
        API --> ROUTES{Rotas da API}
        
        ROUTES --> R1["/api/stocks<br/>GET /popular/list<br/>GET /:symbol<br/>GET /compare"]
        ROUTES --> R2["/api/predictions<br/>GET /:symbol<br/>POST /batch<br/>GET /history"]
        ROUTES --> R3["/api/ml-health<br/>GET /health/:symbol<br/>GET /drift-report<br/>GET /overview<br/>GET /data-quality"]
        ROUTES --> R4["/api/monitoring<br/>GET /<br/>GET /metrics"]
        ROUTES --> R5["/ws<br/>WebSocket real-time"]
        ROUTES --> R6["/metrics<br/>Prometheus format"]
        
        API --> MIDDLEWARE[Middleware<br/>- CORS<br/>- Request timing<br/>- Error handling<br/>- Metrics logging]
    end
    
    subgraph "SERVICES - Lógica de Negócio"
        S1[StockService<br/>- Dados Yahoo Finance<br/>- Cache inteligente<br/>- Indicadores técnicos]
        
        S2[ModelService<br/>- Gerencia modelos LSTM<br/>- HuggingFace Hub<br/>- Cache de modelos<br/>- Fallback BASE]
        
        S3[MLHealthMonitoring<br/>- Feature drift detection<br/>- Prediction analysis<br/>- Data quality checks<br/>- Health scoring 0-100]
        
        S4[MonitoringService<br/>- Coleta métricas<br/>- Request tracking<br/>- System metrics<br/>- Performance KPIs]
        
        S5[PrometheusMetrics<br/>- Counter, Gauge, Histogram<br/>- Labels por símbolo<br/>- Formato Prometheus]
        
        S6[DatabaseService<br/>- PostgreSQL<br/>- Predictions storage<br/>- Model metrics<br/>- Training logs]
        
        S7[EvaluationService<br/>- MAPE calculation<br/>- Model comparison<br/>- Performance tracking]
    end
    
    subgraph "MODELOS ML"
        M1[LSTMPredictor<br/>Original Architecture<br/>- 2 camadas LSTM<br/>- Dropout 0.2<br/>- PyTorch]
        
        M2[ImprovedLSTM<br/>Enhanced Architecture<br/>- 3 camadas LSTM<br/>- Attention mechanism<br/>- Regularização avançada]
        
        M3[Preprocessor<br/>- StandardScaler<br/>- Feature engineering<br/>- Sequencing<br/>- Normalização]
        
        M4[Modelo BASE<br/>Genérico para<br/>todas as ações]
        
        M5[Modelos Específicos<br/>AAPL, GOOGL, MSFT<br/>NVDA, TSLA, etc.]
    end
    
    subgraph "PERSISTÊNCIA"
        DB[(PostgreSQL<br/>Railway Cloud)]
        CACHE[Cache em Memória<br/>Modelos carregados<br/>Previsões recentes<br/>Features históricas]
        HUB[HuggingFace Hub<br/>henriquebap/<br/>stock-predictor-lstm]
    end
    
    subgraph "DADOS EXTERNOS"
        YAHOO[Yahoo Finance API<br/>yfinance library<br/>Dados históricos<br/>Preços em tempo real]
    end
    
    subgraph "TREINO & AVALIAÇÃO"
        T1[Trainer<br/>- Training loop<br/>- Validation<br/>- Early stopping]
        
        T2[SmartTrainer<br/>- Hyperparameter tuning<br/>- Grid search<br/>- Auto-optimization]
        
        T3[ImprovedTrainer<br/>- Advanced techniques<br/>- Learning rate scheduler<br/>- Gradient clipping<br/>- Best model selection]
        
        T4[DataLoader<br/>- Batch processing<br/>- Shuffle<br/>- Train/Val split]
    end
    
    subgraph "MONITORAMENTO AVANÇADO"
        MON1[Infrastructure<br/>- CPU, RAM, Disk<br/>- Request rate<br/>- Response time<br/>- Error rate]
        
        MON2[ML Health<br/>- Feature drift Z-score<br/>- Prediction bias<br/>- Data quality score<br/>- Model health 0-100]
        
        MON3[Prometheus<br/>- Time-series metrics<br/>- Histograms<br/>- Counters & Gauges<br/>- Multi-label support]
        
        MON4[Alertas Automáticos<br/>- Drift detection<br/>- Bias warnings<br/>- Quality issues<br/>- Recomendações]
    end
    
    subgraph "DEPLOY - Railway"
        BACK_DEPLOY[Backend Container<br/>Docker<br/>Python 3.10<br/>Auto-deploy on push]
        
        FRONT_DEPLOY[Frontend Container<br/>Docker<br/>Streamlit<br/>Auto-deploy on push]
        
        DB_DEPLOY[PostgreSQL<br/>Managed Database<br/>Railway Cloud]
    end
    
    subgraph "TESTES"
        TEST1[Unit Tests<br/>pytest<br/>- test_model.py<br/>- test_preprocessor.py<br/>- test_data_loader.py<br/>- test_api.py]
        
        TEST2[Integration Tests<br/>- API endpoints<br/>- Model inference<br/>- Database ops]
    end
    
    %% Fluxos principais
    UI -->|HTTP Requests| HTTP
    HTTP --> API
    
    UI -->|WebSocket| WS
    WS --> API
    
    R1 --> S1
    R2 --> S1
    R2 --> S2
    R3 --> S3
    R4 --> S4
    R4 --> S5
    R6 --> S5
    
    S1 --> YAHOO
    S1 --> CACHE
    
    S2 --> M1
    S2 --> M2
    S2 --> M3
    S2 --> M4
    S2 --> M5
    S2 --> HUB
    S2 --> CACHE
    
    S3 --> CACHE
    S3 -.->|Análise| M4
    S3 -.->|Análise| M5
    
    S4 --> CACHE
    S5 --> CACHE
    
    S6 --> DB
    S7 --> DB
    
    T1 --> M1
    T1 --> M3
    T2 --> M1
    T2 --> M3
    T3 --> M2
    T3 --> M3
    
    T1 --> T4
    T2 --> T4
    T3 --> T4
    
    M4 --> HUB
    M5 --> HUB
    
    S4 --> MON1
    S5 --> MON3
    S3 --> MON2
    MON2 --> MON4
    
    API --> BACK_DEPLOY
    UI --> FRONT_DEPLOY
    DB --> DB_DEPLOY
    
    %% Styling
    classDef frontend fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
    classDef backend fill:#11998e,stroke:#38ef7d,stroke-width:2px,color:#fff
    classDef ml fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    classDef data fill:#ff416c,stroke:#ff4b2b,stroke-width:2px,color:#fff
    classDef monitoring fill:#ffd89b,stroke:#19547b,stroke-width:2px,color:#000
    classDef deploy fill:#a8edea,stroke:#fed6e3,stroke-width:2px,color:#000
    
    class UI,PAGE1,PAGE2,TAB1,TAB2,TAB3,TAB4,SIDEBAR frontend
    class API,ROUTES,R1,R2,R3,R4,R5,R6,MIDDLEWARE backend
    class S1,S2,S3,S4,S5,S6,S7 backend
    class M1,M2,M3,M4,M5 ml
    class T1,T2,T3,T4 ml
    class DB,CACHE,HUB,YAHOO data
    class MON1,MON2,MON3,MON4 monitoring
    class BACK_DEPLOY,FRONT_DEPLOY,DB_DEPLOY deploy
```

### Arquitetura dos Modelos ML

```mermaid
graph TB
    subgraph "HuggingFace Hub - henriquebap/stock-predictor-lstm"
        HUB[Repository<br/>30 arquivos totais]
        
        HUB --> BASE[BASE Model<br/>LSTMPredictor Original<br/>2 LSTM layers<br/>MAPE: 41.46%]
        
        HUB --> IMPROVED{ImprovedLSTM Models<br/>3 LSTM Bidirectional<br/>+ Attention Mechanism}
        
        IMPROVED --> M1[AAPL<br/>MAPE: 8.28%]
        IMPROVED --> M2[GOOGL<br/>Otimizado]
        IMPROVED --> M3[MSFT<br/>Otimizado]
        IMPROVED --> M4[AMZN<br/>Otimizado]
        IMPROVED --> M5[META<br/>Otimizado]
        IMPROVED --> M6[NVDA<br/>Otimizado]
        IMPROVED --> M7[TSLA<br/>Otimizado]
        IMPROVED --> M8[JPM<br/>Otimizado]
        IMPROVED --> M9[V<br/>Otimizado]
    end
    
    subgraph "Backend - ModelService"
        MS[ModelService<br/>Gerenciador Inteligente]
        
        MS --> CACHE{Cache em Memória}
        MS --> LOADER{Smart Loader}
        
        LOADER --> L1[Level 1: Modelo Específico]
        LOADER --> L2[Level 2: Modelo BASE Fallback]
        LOADER --> L3[Level 3: Auto-detecção Arquitetura]
        
        CACHE --> WARM[Warm Start<br/>BASE pré-carregado]
        CACHE --> LAZY[Lazy Loading<br/>Sob demanda]
    end
    
    subgraph "Arquiteturas Suportadas"
        A1[LSTMPredictor<br/>Original<br/>---<br/>• 2 layers LSTM<br/>• Unidirecional<br/>• Dropout 0.2<br/>• Hidden: 50]
        
        A2[ImprovedLSTM<br/>Enhanced<br/>---<br/>• 3 layers LSTM<br/>• Bidirectional<br/>• Attention<br/>• Dropout 0.3<br/>• Hidden: 64<br/>• Layer Norm<br/>• Residual]
    end
    
    subgraph "Cada Modelo Inclui"
        FILES[3 Arquivos por Modelo]
        FILES --> F1[lstm_model_SYMBOL.pth<br/>Pesos treinados PyTorch]
        FILES --> F2[scaler_SYMBOL.pkl<br/>StandardScaler treinado]
        FILES --> F3[metadata_SYMBOL.json<br/>Métricas + Hiperparâmetros]
    end
    
    subgraph "Fluxo de Inferência"
        REQ[Request<br/>/api/predictions/AAPL]
        
        REQ --> CHECK{Está no<br/>Cache?}
        CHECK -->|Sim| USE_CACHE[Usa Modelo<br/>do Cache]
        CHECK -->|Não| DOWNLOAD
        
        DOWNLOAD[Download do Hub]
        DOWNLOAD --> TRY1{Modelo<br/>Específico<br/>Existe?}
        
        TRY1 -->|Sim| LOAD_SPEC[Carrega<br/>lstm_model_AAPL.pth]
        TRY1 -->|Não| LOAD_BASE[Carrega<br/>lstm_model_BASE.pth]
        
        LOAD_SPEC --> DETECT{Auto-detecção<br/>Arquitetura}
        LOAD_BASE --> DETECT
        
        DETECT --> TRY_IMPROVED[Tenta<br/>ImprovedLSTM]
        TRY_IMPROVED -->|Sucesso| LOADED_IMP[Carregado]
        TRY_IMPROVED -->|Falha| TRY_ORIG[Tenta<br/>LSTMPredictor]
        TRY_ORIG --> LOADED_ORIG[Carregado]
        
        LOADED_IMP --> SAVE_CACHE[Salva no Cache]
        LOADED_ORIG --> SAVE_CACHE
        USE_CACHE --> PREDICT
        SAVE_CACHE --> PREDICT[Faz Previsão]
        
        PREDICT --> RESPONSE[Retorna JSON]
    end
    
    subgraph "Uso em Produção"
        PROD[Railway Cloud]
        
        PROD --> STARTUP[Startup<br/>Pré-carrega BASE]
        PROD --> RUNTIME[Runtime<br/>Lazy load outros]
        
        STARTUP --> FAST1[Primeira requisição<br/>BASE: ~100ms]
        RUNTIME --> FAST2[Primeira requisição<br/>Específico: ~2s download]
        RUNTIME --> FAST3[Próximas requisições<br/>Cache: ~50ms]
    end
    
    subgraph "Modelo Destaque"
        BEST[AAPL - Apple<br/>---<br/>MAPE: 8.28%<br/>Melhor Performance<br/>---<br/>ImprovedLSTM<br/>3 layers bidirectional<br/>Attention mechanism<br/>Early stopped: epoch 17]
    end
    
    %% Conexões principais
    HUB -.->|Download| MS
    MS -.->|Usa| A1
    MS -.->|Usa| A2
    BASE -.->|Usa| A1
    M1 -.->|Usa| A2
    M2 -.->|Usa| A2
    M3 -.->|Usa| A2
    M4 -.->|Usa| A2
    M5 -.->|Usa| A2
    M6 -.->|Usa| A2
    M7 -.->|Usa| A2
    M8 -.->|Usa| A2
    M9 -.->|Usa| A2
    
    MS --> PROD
    M1 -.->|É| BEST
    
    %% Styling
    classDef hub fill:#ffd89b,stroke:#19547b,stroke-width:3px,color:#000
    classDef base fill:#667eea,stroke:#764ba2,stroke-width:2px,color:#fff
    classDef improved fill:#11998e,stroke:#38ef7d,stroke-width:2px,color:#fff
    classDef service fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    classDef arch fill:#ff416c,stroke:#ff4b2b,stroke-width:2px,color:#fff
    classDef best fill:#FFD700,stroke:#FF8C00,stroke-width:4px,color:#000
    classDef flow fill:#a8edea,stroke:#fed6e3,stroke-width:2px,color:#000
    
    class HUB hub
    class BASE base
    class IMPROVED,M1,M2,M3,M4,M5,M6,M7,M8,M9 improved
    class MS,CACHE,LOADER,L1,L2,L3,WARM,LAZY service
    class A1,A2 arch
    class BEST best
    class REQ,CHECK,USE_CACHE,DOWNLOAD,TRY1,LOAD_SPEC,LOAD_BASE,DETECT,TRY_IMPROVED,TRY_ORIG,LOADED_IMP,LOADED_ORIG,SAVE_CACHE,PREDICT,RESPONSE flow
    class PROD,STARTUP,RUNTIME,FAST1,FAST2,FAST3 flow
```

## Funcionalidades

### Core Features

- **Previsão de Preços**: Previsão de preços de fechamento de ações usando LSTM
- **Múltiplas Ações**: Suporte para previsão de múltiplas ações
- **Previsões em Lote**: API endpoint para previsões batch
- **Dados Históricos**: Acesso a dados históricos via API
- **Treinamento Automático**: Pipeline completo de treinamento com validação
- **Retreinamento**: Endpoint para retreinar modelos sob demanda

### API & Monitoring

- **API RESTful**: FastAPI com documentação automática (Swagger/OpenAPI)
- **Monitoramento**: Métricas Prometheus e dashboard de monitoramento
- **Logging Estruturado**: Logs detalhados com Loguru
- **Health Checks**: Endpoints de saúde da aplicação
- **CORS**: Configuração CORS para integração frontend

### DevOps

- **Containerização**: Docker e Docker Compose
- **CI/CD**: GitHub Actions
- **Testes**: Suite de testes com pytest
- **Agendamento**: Scripts para treinamento agendado (cron)

## Instalação

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

Crie um arquivo `.env` na raiz do projeto:

```bash
# Backend
DATABASE_URL=postgresql://user:password@localhost:5432/stockdb
PORT=8000
HF_TOKEN=your_huggingface_token  # Opcional, para upload de modelos

# Frontend
API_URL=http://localhost:8000
PORT=8501

# Opcional
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Instalação com Docker

```bash
docker-compose up --build
```

A API estará disponível em `http://localhost:8000`

## Uso

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
# Previsão para AAPL
curl https://previsaoacoes-back-production.up.railway.app/api/predictions/AAPL

# Dados de uma ação
curl https://previsaoacoes-back-production.up.railway.app/api/stocks/GOOGL

# Lista de ações populares
curl https://previsaoacoes-back-production.up.railway.app/api/stocks/popular/list

# Health de um modelo
curl https://previsaoacoes-back-production.up.railway.app/api/ml-health/health/AAPL

# Métricas de monitoramento
curl https://previsaoacoes-back-production.up.railway.app/api/monitoring
```

## API Endpoints

### Aplicação em Produção

**Backend API**: `https://previsaoacoes-back-production.up.railway.app`  
**Frontend Dashboard**: `https://stock-pred.up.railway.app`  
**Documentação Interativa**: `https://previsaoacoes-back-production.up.railway.app/docs`

### Stocks (Dados de Ações)

- `GET /api/stocks/popular/list` - Lista ações populares (AAPL, GOOGL, MSFT, etc)
- `GET /api/stocks/{symbol}` - Dados históricos e preço atual de uma ação
- `GET /api/stocks/compare` - Comparação entre múltiplas ações

### Predictions (Previsões)

- `GET /api/predictions/{symbol}` - Previsão de preço para uma ação específica
- `POST /api/predictions/batch` - Previsões em lote para múltiplas ações
- `GET /api/predictions/history` - Histórico de previsões

**Exemplo de uso:**
```bash
curl https://previsaoacoes-back-production.up.railway.app/api/predictions/AAPL
```

### ML Health (Saúde dos Modelos)

- `GET /api/ml-health/health/{symbol}` - Health score (0-100) do modelo
- `GET /api/ml-health/drift-report` - Relatório de data drift
- `GET /api/ml-health/overview` - Visão geral de todos os modelos
- `GET /api/ml-health/data-quality` - Qualidade dos dados de entrada
- `GET /api/ml-health/prediction-distribution/{symbol}` - Análise de distribuição

### Monitoring (Monitoramento)

- `GET /api/monitoring` - Métricas gerais da API
- `GET /metrics` - Métricas em formato Prometheus

### WebSocket

- `WS /ws` - Conexão WebSocket para atualizações em tempo real

## Treinamento de Modelos

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

## Deploy

### Aplicação em Produção

O projeto está deployado no **Railway** com arquitetura de microserviços:

**URLs de Acesso:**
- **Frontend (Streamlit)**: https://stock-pred.up.railway.app
- **Backend (FastAPI)**: https://previsaoacoes-back-production.up.railway.app
- **API Docs**: https://previsaoacoes-back-production.up.railway.app/docs
- **PostgreSQL**: Managed database no Railway Cloud

### Modelos ML (HuggingFace Hub)

Os modelos LSTM treinados estão hospedados no HuggingFace Hub:

**Repository**: https://huggingface.co/henriquebap/stock-predictor-lstm

**Modelos Disponíveis:**
- `BASE` - Modelo genérico (MAPE: 41.46%)
- `AAPL` - Apple (MAPE: 8.28% - Melhor performance)
- `GOOGL` - Google
- `MSFT` - Microsoft
- `AMZN` - Amazon
- `META` - Meta/Facebook
- `NVDA` - NVIDIA
- `TSLA` - Tesla
- `JPM` - JP Morgan
- `V` - Visa

Total: **11 modelos** (1 BASE + 10 específicos)

### Arquitetura de Deploy

```
Railway Cloud
├── Backend Container (FastAPI)
│   ├── Python 3.10
│   ├── Auto-deploy on push
│   └── Download modelos do HuggingFace Hub
├── Frontend Container (Streamlit)
│   ├── Dashboard interativo
│   └── Monitoramento em tempo real
└── PostgreSQL Database
    ├── Predictions storage
    ├── Model metrics
    └── Training logs
```

### Como Replicar o Deploy no Railway

**1. Backend (FastAPI)**

```bash
# No Railway, criar novo projeto a partir do GitHub
# Configurar:
Root Directory: railway_app/backend
Build Command: (automático - Dockerfile)
Start Command: (automático - Dockerfile)

# Variáveis de Ambiente:
DATABASE_URL=${{Postgres.DATABASE_URL}}  # Auto-gerado pelo Railway
PORT=8000
PYTHONUNBUFFERED=1
```

**2. Frontend (Streamlit)**

```bash
# Criar segundo serviço no mesmo projeto
Root Directory: railway_app/frontend
Build Command: (automático - Dockerfile)
Start Command: (automático - Dockerfile)

# Variáveis de Ambiente:
API_URL=https://seu-backend.up.railway.app
PORT=8501
```

**3. PostgreSQL**

```bash
# Adicionar PostgreSQL do Railway Marketplace
# Conecta automaticamente ao backend via ${{Postgres.DATABASE_URL}}
```

**4. Deploy Automático**

- Push para branch `main` → Deploy automático
- Railway faz build dos Dockerfiles
- Modelos são baixados do HuggingFace Hub na primeira execução
- URLs geradas automaticamente pelo Railway

### Configuração do HuggingFace Hub

Para fazer upload de modelos treinados:

```bash
# Instalar HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload de modelo
python scripts/upload_to_hub.py --model models/lstm_model_AAPL.pth --symbol AAPL
```

Os modelos são automaticamente baixados pelo backend quando necessário.

### Docker Local

```bash
# Backend
cd railway_app/backend
docker build -t stock-backend .
docker run -p 8000:8000 stock-backend

# Frontend
cd railway_app/frontend
docker build -t stock-frontend .
docker run -p 8501:8501 -e API_URL=http://localhost:8000 stock-frontend
```

### Docker Compose (Desenvolvimento Local)

```bash
docker-compose up --build
```

Acesse:
- Backend: http://localhost:8000
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

## Testes

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

## Monitoramento

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

## Estrutura do Projeto

```
previsao_acoes/
├── railway_app/                    # Aplicação em produção (Railway)
│   ├── backend/                    # Backend FastAPI
│   │   ├── main.py                 # Entry point da API
│   │   ├── routes/                 # API routes
│   │   │   ├── predictions.py      # Endpoints de previsão
│   │   │   ├── stocks.py           # Endpoints de ações
│   │   │   ├── ml_health.py        # ML Health monitoring
│   │   │   └── websocket.py        # WebSocket real-time
│   │   ├── services/               # Lógica de negócio
│   │   │   ├── model_service.py    # Gerenciamento de modelos
│   │   │   ├── stock_service.py    # Serviço de dados
│   │   │   ├── ml_health.py        # ML Health monitoring
│   │   │   ├── monitoring.py       # Métricas de sistema
│   │   │   └── prometheus_metrics.py
│   │   ├── core/                   # Modelos ML
│   │   │   ├── lstm_model.py       # LSTM original
│   │   │   ├── improved_lstm.py    # LSTM com Attention
│   │   │   └── preprocessor.py     # Preprocessamento
│   │   ├── database/               # PostgreSQL
│   │   │   └── service.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── frontend/                   # Frontend Streamlit
│       ├── app.py                  # Dashboard principal
│       ├── components/
│       │   └── sidebar.py          # Componentes UI
│       ├── Dockerfile
│       └── requirements.txt
├── src/                            # Código de treino/desenvolvimento
│   ├── training/                   # Pipeline de treinamento
│   │   ├── trainer.py              # Trainer básico
│   │   ├── smart_trainer.py        # Trainer com tuning
│   │   └── improved_trainer.py     # Trainer avançado
│   ├── data/                       # Data handling
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   └── utils/                      # Utilities
├── tests/                          # Test suite
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_preprocessor.py
│   └── test_data_loader.py
├── scripts/                        # Scripts de treino
│   └── train_model.py
├── models/                         # Modelos salvos localmente
│   └── hub_cache/                  # Cache do HuggingFace Hub
├── data/                           # Data storage
├── logs/                           # Application logs
├── .github/workflows/              # CI/CD
├── docker-compose.yml
└── README.md
```

### Separação Backend/Frontend

O projeto usa arquitetura de **microserviços separados**:

- **Backend (FastAPI)**: API REST, modelos ML, banco de dados
- **Frontend (Streamlit)**: Dashboard interativo, visualizações
- **Comunicação**: HTTP REST + WebSocket
- **Deploy**: Containers Docker independentes no Railway


## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Autores

**Henrique Baptista**
- GitHub: [@henriquebap](https://github.com/henriquebap)
- LinkedIn: [henrique-baptista777](https://www.linkedin.com/in/henrique-baptista777/)

**Felipe Araujo De Almeida**
- GitHub: [@Felpz2212](https://github.com/Felpz2212)

**Carlos Eduardo Cheim**
- GitHub: [@CECH-Carlos](https://github.com/CECH-Carlos)

## Agradecimentos

- FIAP Pós-Tech MLET
- Tech Challenge Fase 4
- Comunidade Python/PyTorch
- Colaboradores e revisores

## Referências

- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Yahoo Finance API](https://github.com/ranaroussi/yfinance)

---

**Nota**: Este é um projeto educacional. Não use para decisões reais de investimento sem análise adicional e consultoria profissional.

