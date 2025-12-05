# Guia de Deployment

Este documento fornece instruções detalhadas para fazer deploy da API de previsão de ações em diferentes plataformas.

## 📋 Índice

- [Railway](#railway)
- [HuggingFace Spaces](#huggingface-spaces)
- [Docker](#docker)
- [Supabase (Database)](#supabase-database)

## 🚂 Railway

Railway é uma plataforma de deploy simples e poderosa que suporta Docker e possui tier gratuito.

### Pré-requisitos

- Conta no [Railway](https://railway.app/)
- Repositório Git com o projeto

### Passo a Passo

1. **Instale o Railway CLI**

```bash
npm i -g @railway/cli
```

2. **Faça login no Railway**

```bash
railway login
```

3. **Inicialize o projeto**

```bash
railway init
```

4. **Configure variáveis de ambiente**

```bash
# Via CLI
railway variables set DATABASE_URL=postgresql://...
railway variables set DEFAULT_STOCK_SYMBOL=AAPL
railway variables set LSTM_EPOCHS=50

# Ou via Dashboard do Railway
# https://railway.app/project/your-project/variables
```

5. **Deploy**

```bash
# Deploy manual
railway up

# Deploy via GitHub (recomendado)
# Conecte seu repo no dashboard do Railway
# Cada push no main vai fazer deploy automaticamente
```

### railway.json (Opcional)

Crie um arquivo `railway.json` na raiz do projeto:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10,
    "healthcheckPath": "/api/v1/health",
    "healthcheckTimeout": 300
  }
}
```

### Variáveis de Ambiente Necessárias

```bash
# Banco de Dados (se usar Supabase)
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2

# Model Configuration
DEFAULT_STOCK_SYMBOL=AAPL
DEFAULT_START_DATE=2018-01-01
DEFAULT_END_DATE=2024-12-31
LSTM_SEQUENCE_LENGTH=60
LSTM_EPOCHS=50
LSTM_BATCH_SIZE=32
LSTM_LEARNING_RATE=0.001

# Monitoring
LOG_LEVEL=INFO
```

### Volumes Persistentes

Railway suporta volumes para persistir dados:

```bash
# Crie volume para modelos
railway volume create models --mount /app/models

# Crie volume para dados
railway volume create data --mount /app/data
```

### Domínio Customizado

```bash
# Via CLI
railway domain

# Ou configure no dashboard
# Settings > Networking > Custom Domain
```

### Monitoramento

Acesse logs em tempo real:

```bash
railway logs
```

## 🤗 HuggingFace Spaces

HuggingFace Spaces permite hospedar demos interativos gratuitamente.

### Criar Interface Gradio

Crie um arquivo `app_gradio.py`:

```python
import gradio as gr
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# URL da sua API no Railway
API_URL = "https://your-api.railway.app"

def predict_stock(symbol, days_ahead):
    """Fazer previsão via API."""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json={"symbol": symbol, "days_ahead": days_ahead}
        )
        
        if response.status_code == 200:
            data = response.json()
            return f"""
            **Previsão para {symbol}**
            
            Preço Atual: ${data['current_price']:.2f}
            Preço Previsto: ${data['predicted_price']:.2f}
            Data da Previsão: {data['prediction_date']}
            Variação: {((data['predicted_price'] - data['current_price']) / data['current_price'] * 100):.2f}%
            """
        else:
            return f"Erro: {response.json().get('detail', 'Erro desconhecido')}"
    except Exception as e:
        return f"Erro de conexão: {str(e)}"

def get_historical_data(symbol):
    """Obter dados históricos via API."""
    try:
        response = requests.get(
            f"{API_URL}/api/v1/stocks/{symbol}/historical",
            params={"limit": 365}
        )
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            
            # Criar gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['close'],
                mode='lines',
                name='Preço de Fechamento'
            ))
            fig.update_layout(
                title=f'Histórico de Preços - {symbol}',
                xaxis_title='Data',
                yaxis_title='Preço (USD)',
                hovermode='x'
            )
            
            return fig
        else:
            return None
    except Exception as e:
        print(f"Erro: {str(e)}")
        return None

# Interface Gradio
with gr.Blocks(title="Stock Price Predictor") as demo:
    gr.Markdown("""
    # 📈 Stock Price Predictor - LSTM
    
    Previsão de preços de ações usando redes neurais LSTM.
    Desenvolvido como parte do Tech Challenge Fase 4 - FIAP.
    """)
    
    with gr.Tab("Previsão"):
        with gr.Row():
            with gr.Column():
                symbol_input = gr.Textbox(
                    label="Símbolo da Ação",
                    placeholder="Ex: AAPL, GOOGL, MSFT",
                    value="AAPL"
                )
                days_input = gr.Slider(
                    minimum=1,
                    maximum=30,
                    step=1,
                    value=1,
                    label="Dias à frente"
                )
                predict_btn = gr.Button("Fazer Previsão", variant="primary")
            
            with gr.Column():
                prediction_output = gr.Markdown()
        
        predict_btn.click(
            predict_stock,
            inputs=[symbol_input, days_input],
            outputs=prediction_output
        )
    
    with gr.Tab("Histórico"):
        with gr.Row():
            symbol_hist = gr.Textbox(
                label="Símbolo da Ação",
                placeholder="Ex: AAPL",
                value="AAPL"
            )
            hist_btn = gr.Button("Carregar Histórico", variant="primary")
        
        chart_output = gr.Plot()
        
        hist_btn.click(
            get_historical_data,
            inputs=symbol_hist,
            outputs=chart_output
        )
    
    gr.Markdown("""
    ---
    **Nota**: Este é um projeto educacional. Não use para decisões reais de investimento.
    
    [GitHub](https://github.com/your-username/previsao_acoes) | [API Docs](https://your-api.railway.app/docs)
    """)

if __name__ == "__main__":
    demo.launch()
```

### Deploy no HuggingFace

1. **Crie uma conta no [HuggingFace](https://huggingface.co/)**

2. **Crie um novo Space**
   - Vá em https://huggingface.co/spaces
   - Clique em "Create new Space"
   - Escolha "Gradio" como SDK
   - Nomeie seu space (ex: `stock-prediction-demo`)

3. **Configure o repositório**

```bash
# Clone seu space
git clone https://huggingface.co/spaces/your-username/stock-prediction-demo
cd stock-prediction-demo

# Copie arquivos necessários
cp app_gradio.py app.py
```

4. **Crie requirements.txt para o Space**

```txt
gradio==4.0.0
requests==2.31.0
pandas==2.0.3
plotly==5.17.0
```

5. **Faça push**

```bash
git add .
git commit -m "Add Gradio interface"
git push
```

O deploy é automático! Seu app estará disponível em:
`https://huggingface.co/spaces/your-username/stock-prediction-demo`

## 🐳 Docker

### Build Local

```bash
docker build -t stock-prediction-api:latest .
```

### Run

```bash
docker run -d \
  --name stock-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e DATABASE_URL=postgresql://... \
  stock-prediction-api:latest
```

### Docker Compose

```bash
# Start
docker-compose up -d

# Logs
docker-compose logs -f

# Stop
docker-compose down
```

### Docker Hub

```bash
# Tag
docker tag stock-prediction-api:latest your-username/stock-prediction-api:latest

# Push
docker push your-username/stock-prediction-api:latest
```

## 🗄️ Supabase (Database)

Supabase oferece PostgreSQL gerenciado gratuitamente.

### Setup

1. **Crie um projeto no [Supabase](https://supabase.com/)**

2. **Obtenha a connection string**
   - Settings > Database > Connection string
   - Modo: "URI"

3. **Configure no Railway**

```bash
railway variables set DATABASE_URL="postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres"
```

### Schema SQL

```sql
-- Tabela para dados históricos
CREATE TABLE stocks_historical (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_symbol_timestamp ON stocks_historical(symbol, timestamp);

-- Tabela para previsões
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_price DECIMAL(10, 2),
    actual_price DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabela para metadados dos modelos
CREATE TABLE model_metadata (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    trained_at TIMESTAMP,
    rmse DECIMAL(10, 4),
    mae DECIMAL(10, 4),
    mape DECIMAL(10, 4),
    r2 DECIMAL(10, 4),
    model_version VARCHAR(50)
);
```

## 🔄 CI/CD com GitHub Actions

O workflow já está configurado em `.github/workflows/ci-cd.yml`.

### Deploy Automático no Railway

1. **Obtenha o token do Railway**

```bash
railway whoami
```

2. **Configure como Secret no GitHub**
   - Vá em Settings > Secrets > Actions
   - Adicione `RAILWAY_TOKEN`

3. **Descomente o job de deploy** no workflow

O deploy será feito automaticamente em cada push no main.

## 🔐 Segurança

### Variáveis Sensíveis

- **NUNCA** commite arquivos `.env` ou secrets
- Use variáveis de ambiente da plataforma
- Rotacione credentials regularmente

### Rate Limiting

Considere adicionar rate limiting na API:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/predict")
@limiter.limit("5/minute")
async def predict():
    ...
```

## 📊 Monitoramento Pós-Deploy

### Health Checks

Configure health checks na sua plataforma:
- **Railway**: Automático via `/api/v1/health`
- **Docker**: HEALTHCHECK no Dockerfile

### Logging

- Railway: `railway logs`
- Docker: `docker logs -f container_name`

### Alertas

Configure alertas para:
- API downtime
- Erros de modelo
- Uso excessivo de recursos

## 🆘 Troubleshooting

### Erro: Model not found

- Certifique-se de treinar o modelo antes de fazer previsões
- Verifique se os volumes estão montados corretamente

### Erro: Out of memory

- Reduza `LSTM_BATCH_SIZE`
- Aumente memória no Railway (plan upgrade)
- Use workers menores

### Erro: Database connection

- Verifique `DATABASE_URL`
- Certifique-se que o IP está whitelisted (Supabase)

---

Para mais ajuda, abra uma issue no [GitHub](https://github.com/your-username/previsao_acoes/issues).

