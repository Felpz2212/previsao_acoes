# Stock Predictor - Railway App

Aplicacao completa de previsao de precos de acoes usando LSTM, FastAPI e Streamlit.

## Arquitetura

```
[Streamlit Frontend] <---> [FastAPI Backend] <---> [LSTM Model]
     (Railway)              (Railway)           (Local + HF Hub)
         |                      |
         v                      v
    [Plotly Charts]       [Yahoo Finance]
    [WebSocket]           [Model Cache]
```

## Estrutura

```
railway_app/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── routes/
│   │   ├── predictions.py   # Previsoes LSTM
│   │   ├── stocks.py        # Dados de acoes
│   │   └── websocket.py     # Precos realtime
│   ├── services/
│   │   ├── model_service.py # Gerencia LSTM
│   │   └── stock_service.py # Yahoo Finance
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py               # Streamlit app
│   ├── components/
│   │   ├── charts.py        # Graficos Plotly
│   │   ├── sidebar.py       # Navegacao
│   │   └── predictions.py   # Card de previsao
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── railway.toml
```

## Rodando Localmente

### 1. Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Acesse: http://localhost:8000/docs

### 2. Frontend (Streamlit)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Acesse: http://localhost:8501

### 3. Com Docker Compose

```bash
cd railway_app
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

## Deploy no Railway

### Opcao 1: Deploy via GitHub

1. Conecte seu repositorio ao Railway
2. Crie dois servicos:
   - **backend**: Aponte para `railway_app/backend`
   - **frontend**: Aponte para `railway_app/frontend`
3. Configure variaveis:
   - Backend: `HF_TOKEN`
   - Frontend: `API_URL` (URL do backend no Railway)

### Opcao 2: Deploy via CLI

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy backend
cd backend
railway up

# Deploy frontend  
cd ../frontend
railway up
```

## API Endpoints

### Stocks

- `GET /api/stocks/{symbol}` - Dados historicos
- `GET /api/stocks/compare/multiple?symbols=AAPL,GOOGL` - Comparar acoes
- `GET /api/stocks/popular/list` - Acoes populares

### Predictions

- `GET /api/predictions/{symbol}` - Previsao LSTM
- `GET /api/predictions/history/recent` - Historico

### WebSocket

- `WS /ws/prices` - Precos em tempo real

## Features

- **Graficos Candlestick** com medias moveis e volume
- **Previsoes LSTM** com indicadores tecnicos
- **Comparacao** de multiplas acoes (base 100)
- **Busca inteligente** por nome ou ticker
- **WebSocket** para atualizacao em tempo real
- **Historico** de previsoes

## Tech Stack

- **Backend**: FastAPI, PyTorch, yfinance, HuggingFace Hub
- **Frontend**: Streamlit, Plotly
- **Deploy**: Docker, Railway
- **Modelo**: LSTM (HuggingFace Model Hub)

## Variaveis de Ambiente

| Variavel | Descricao | Padrao |
|----------|-----------|--------|
| HF_TOKEN | Token HuggingFace | - |
| API_URL | URL do backend | http://localhost:8000 |

## Disclaimer

Este projeto e educacional. NAO use para investimentos reais!

