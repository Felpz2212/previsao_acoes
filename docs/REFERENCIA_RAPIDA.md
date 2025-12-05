# ⚡ Referência Rápida

> Cheat sheet com comandos e conceitos essenciais

---

## 🚀 Comandos Essenciais

### Setup Inicial

```bash
# Clone e setup
git clone https://github.com/your-username/previsao_acoes.git
cd previsao_acoes
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Treinamento

```bash
# Básico
python scripts/train_model.py AAPL

# Com opções
python scripts/train_model.py GOOGL \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --epochs 100 \
  --batch-size 64
```

### API

```bash
# Desenvolvimento
uvicorn src.api.main:app --reload

# Produção
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

```bash
# Build e run
docker-compose up --build

# Apenas build
docker build -t stock-api .

# Run manual
docker run -d -p 8000:8000 --name stock-api stock-api
```

### Testes

```bash
# Todos os testes
pytest

# Com cobertura
pytest --cov=src tests/

# Específico
pytest tests/test_api.py::test_prediction -v
```

---

## 🌐 Endpoints da API

| Método | Endpoint | Descrição | Exemplo |
|--------|----------|-----------|---------|
| **POST** | `/api/v1/predict` | Previsão simples | `{"symbol": "AAPL", "days_ahead": 1}` |
| **POST** | `/api/v1/predict/batch` | Previsões em lote | `{"symbols": ["AAPL", "GOOGL"], "days_ahead": 1}` |
| **GET** | `/api/v1/stocks/{symbol}/historical` | Dados históricos | `?limit=365` |
| **GET** | `/api/v1/stocks/{symbol}/latest` | Preço atual | - |
| **GET** | `/api/v1/stocks/available` | Ações disponíveis | - |
| **POST** | `/api/v1/models/train` | Treinar modelo | `{"symbol": "AAPL", "start_date": "2020-01-01"}` |
| **GET** | `/api/v1/models/status` | Status modelos | - |
| **GET** | `/api/v1/models/{symbol}/performance` | Métricas | - |
| **GET** | `/api/v1/health` | Health check | - |
| **GET** | `/api/v1/metrics` | Métricas API | - |

### Exemplos cURL

```bash
# Previsão
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 1}'

# Dados históricos
curl "http://localhost:8000/api/v1/stocks/AAPL/historical?limit=100"

# Status
curl "http://localhost:8000/api/v1/models/status"
```

---

## 📊 Estrutura de Arquivos

```
previsao_acoes/
├── src/                    # Código-fonte
│   ├── api/               # FastAPI
│   │   ├── main.py       # Entry point
│   │   ├── schemas.py    # Pydantic models
│   │   └── routes/       # Endpoints
│   ├── data/             # Data handling
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/           # ML models
│   │   └── lstm_model.py
│   ├── training/         # Training
│   │   └── trainer.py
│   └── utils/            # Utilities
│       ├── logger.py
│       └── monitoring.py
├── tests/                 # Testes
├── scripts/              # Scripts CLI
│   └── train_model.py
├── config/               # Configurações
│   └── settings.py
├── models/               # Modelos salvos
├── data/                 # Dados
├── logs/                 # Logs
├── notebooks/            # Jupyter
├── Dockerfile            # Container
├── docker-compose.yml    # Orquestração
└── requirements.txt      # Dependências
```

---

## 🎯 Hiperparâmetros Principais

### Configuração Padrão

```python
# config/settings.py
LSTM_SEQUENCE_LENGTH = 60    # Janela de 60 dias
LSTM_EPOCHS = 50             # 50 épocas
LSTM_BATCH_SIZE = 32         # Batch de 32
LSTM_LEARNING_RATE = 0.001   # LR padrão Adam
LSTM_HIDDEN_SIZE = 50        # 50 neurônios
LSTM_NUM_LAYERS = 2          # 2 camadas
LSTM_DROPOUT = 0.2           # 20% dropout
```

### Como Ajustar

```bash
# Via CLI
python scripts/train_model.py AAPL --epochs 100 --batch-size 64

# Via variáveis de ambiente
export LSTM_EPOCHS=100
export LSTM_BATCH_SIZE=64
python scripts/train_model.py AAPL
```

---

## 📈 Features do Modelo

### 16 Features Criadas

| # | Feature | Tipo | Cálculo |
|---|---------|------|---------|
| 1 | `open` | Preço | Valor bruto |
| 2 | `high` | Preço | Valor bruto |
| 3 | `low` | Preço | Valor bruto |
| 4 | `close` | Preço | Valor bruto (target) |
| 5 | `volume` | Volume | Valor bruto |
| 6 | `price_change` | Variação | `close.pct_change()` |
| 7 | `high_low_pct` | Variação | `(high - low) / low` |
| 8 | `close_open_pct` | Variação | `(close - open) / open` |
| 9 | `ma_7` | Média Móvel | `close.rolling(7).mean()` |
| 10 | `ma_30` | Média Móvel | `close.rolling(30).mean()` |
| 11 | `ma_90` | Média Móvel | `close.rolling(90).mean()` |
| 12 | `volatility_7` | Volatilidade | `close.rolling(7).std()` |
| 13 | `volatility_30` | Volatilidade | `close.rolling(30).std()` |
| 14 | `volume_change` | Volume | `volume.pct_change()` |
| 15 | `volume_ma_7` | Volume | `volume.rolling(7).mean()` |
| 16 | `momentum` | Momentum | `close - close.shift(4)` |

---

## 📊 Métricas de Avaliação

### Interpretação Rápida

| Métrica | Fórmula | Bom Valor | Interpretação |
|---------|---------|-----------|---------------|
| **RMSE** | `√(Σ(pred-real)²/n)` | < 5% preço | Erro em $ |
| **MAE** | `Σ|pred-real|/n` | < 3% preço | Erro médio |
| **MAPE** | `Σ|real-pred|/|real|/n×100` | < 10% | Erro % |
| **R²** | `1-(SS_res/SS_tot)` | > 0.7 | % variância explicada |
| **Dir Acc** | `#corretos/#total×100` | > 60% | % direção correta |

### Exemplo de Resultado

```
✅ EXCELENTE
RMSE:  3.45    (1.9% do preço médio)
MAE:   2.67    (1.5% do preço médio)
MAPE:  1.89%   (< 10% é excelente)
R²:    0.9567  (explica 95.67%)
Dir:   76.47%  (acerta 3 de 4)
```

---

## 🔧 Variáveis de Ambiente

### Principais Configurações

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Model
DEFAULT_STOCK_SYMBOL=AAPL
DEFAULT_START_DATE=2018-01-01
DEFAULT_END_DATE=2024-12-31

# LSTM
LSTM_SEQUENCE_LENGTH=60
LSTM_EPOCHS=50
LSTM_BATCH_SIZE=32
LSTM_LEARNING_RATE=0.001

# Monitoring
LOG_LEVEL=INFO

# Database (opcional)
DATABASE_URL=postgresql://user:pass@host:5432/db
```

### Arquivo .env

```bash
# Criar arquivo .env
cat > .env << EOF
API_HOST=0.0.0.0
API_PORT=8000
DEFAULT_STOCK_SYMBOL=AAPL
LSTM_EPOCHS=50
LOG_LEVEL=INFO
EOF
```

---

## 🐛 Troubleshooting Rápido

| Erro | Causa | Solução |
|------|-------|---------|
| Model not found | Modelo não treinado | `python scripts/train_model.py AAPL` |
| Port already in use | Porta 8000 ocupada | `uvicorn src.api.main:app --port 8001` |
| Module not found | Deps não instaladas | `pip install -r requirements.txt` |
| Insufficient data | Período muito curto | Usar `--start-date` mais antigo |
| CUDA out of memory | Batch size muito grande | `--batch-size 16` |
| Bad predictions | Modelo não convergiu | Treinar com mais epochs ou dados |

---

## 📦 Dependências Principais

### Core

```
python==3.10
torch==2.0.1
fastapi==0.104.1
uvicorn==0.24.0
yfinance==0.2.28
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
```

### Instalação

```bash
# Básico
pip install torch fastapi uvicorn yfinance pandas numpy scikit-learn

# Completo
pip install -r requirements.txt
```

---

## 🚢 Deploy Rápido

### Railway

```bash
# 1. Install CLI
npm i -g @railway/cli

# 2. Login
railway login

# 3. Init
railway init

# 4. Deploy
railway up
```

### Docker Local

```bash
# Build
docker build -t stock-api .

# Run
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  stock-api
```

### HuggingFace Spaces

```bash
# 1. Create Space (Gradio SDK)
# 2. Clone
git clone https://huggingface.co/spaces/user/space-name

# 3. Copy files
cp app_gradio.py space-name/app.py

# 4. Push
cd space-name
git add . && git commit -m "Add app" && git push
```

---

## 💻 Python API Client

### Exemplo Básico

```python
import requests

API_URL = "http://localhost:8000"

# Previsão
response = requests.post(
    f"{API_URL}/api/v1/predict",
    json={"symbol": "AAPL", "days_ahead": 1}
)
data = response.json()
print(f"Previsão: ${data['predicted_price']:.2f}")

# Histórico
response = requests.get(
    f"{API_URL}/api/v1/stocks/AAPL/historical",
    params={"limit": 100}
)
historical = response.json()
```

### Classe Helper

```python
class StockPredictionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, symbol, days_ahead=1):
        response = requests.post(
            f"{self.base_url}/api/v1/predict",
            json={"symbol": symbol, "days_ahead": days_ahead}
        )
        return response.json()
    
    def get_historical(self, symbol, limit=365):
        response = requests.get(
            f"{self.base_url}/api/v1/stocks/{symbol}/historical",
            params={"limit": limit}
        )
        return response.json()

# Uso
client = StockPredictionClient()
pred = client.predict("AAPL")
```

---

## 📝 Logs e Monitoramento

### Acessar Logs

```bash
# Logs da aplicação
tail -f logs/app_2024-12-02.log

# Apenas erros
tail -f logs/errors_2024-12-02.log

# Filtrar por termo
grep "ERROR" logs/app_2024-12-02.log

# Docker logs
docker logs -f stock-api
```

### Métricas

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Métricas da API
curl http://localhost:8000/api/v1/metrics | jq

# Prometheus format
curl http://localhost:8000/api/v1/metrics/prometheus
```

---

## 🔍 Validação Rápida

### Checklist Após Treinar

```bash
# 1. Verificar se modelo foi criado
ls -lh models/lstm_model_AAPL.pth

# 2. Ver métricas
cat models/metadata_AAPL.json | jq '.metrics'

# 3. Testar previsão
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 1}' | jq

# 4. Verificar performance
curl "http://localhost:8000/api/v1/models/AAPL/performance" | jq
```

---

## 📚 Links Úteis

### Documentação

- **API Local**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health**: http://localhost:8000/api/v1/health

### Referências Externas

- [PyTorch Docs](https://pytorch.org/docs/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Pandas](https://pandas.pydata.org/)

---

## 🎓 Conceitos-Chave

### LSTM

- **Long Short-Term Memory**: RNN que mantém memória de longo prazo
- **Cell State**: Memória que flui pela rede
- **Gates**: Forget, Input, Output gates controlam fluxo de informação

### Time Series

- **Sequence Length**: Janela de histórico (60 dias)
- **Lookback**: Quantos dias usar para prever
- **Horizon**: Quantos dias à frente prever

### ML Engineering

- **Training Loop**: Forward → Loss → Backward → Update
- **Validation**: Dados não vistos para avaliar
- **Test Set**: Dados completamente separados
- **Overfitting**: Memoriza treino, ruim no teste
- **Regularization**: Dropout previne overfitting

---

## ⚡ Atalhos do Sistema

### Aliases Úteis

```bash
# Adicione ao ~/.bashrc ou ~/.zshrc

# Treinar
alias train='python scripts/train_model.py'

# API
alias api='uvicorn src.api.main:app --reload'

# Testes
alias test='pytest -v'

# Docker
alias dup='docker-compose up'
alias ddown='docker-compose down'
alias dlogs='docker-compose logs -f'

# Uso
train AAPL
api
test
```

---

## 🎯 Workflow Típico

### Dia a Dia de Desenvolvimento

```bash
# 1. Ativar ambiente
source venv/bin/activate

# 2. Ver status
git status

# 3. Treinar modelo (se necessário)
python scripts/train_model.py AAPL

# 4. Rodar testes
pytest

# 5. Iniciar API
uvicorn src.api.main:app --reload

# 6. Testar endpoint
curl http://localhost:8000/api/v1/health

# 7. Fazer mudanças...

# 8. Commit
git add .
git commit -m "feat: adiciona nova feature"
git push

# 9. CI/CD roda automaticamente
```

---

## 🔐 Segurança

### Boas Práticas

```bash
# Nunca commitar .env
echo ".env" >> .gitignore

# Rotacionar secrets
railway variables set API_KEY=new_key

# Rate limiting (adicionar no futuro)
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

# HTTPS em produção
# Railway já fornece automaticamente
```

---

## 📊 Performance

### Benchmarks

| Operação | Tempo | Recursos |
|----------|-------|----------|
| Treinamento AAPL | ~15min | CPU: 80%, RAM: 500MB |
| Predição single | ~200ms | CPU: 10%, RAM: 100MB |
| Carga de modelo | ~2s | RAM: +200MB |
| Download dados | ~5s | Network: 1MB |

### Otimizações

```python
# Cache de modelos
from functools import lru_cache

@lru_cache(maxsize=10)
def load_model(symbol):
    return LSTMPredictor.load(get_model_path(symbol))

# Batch predictions
async def predict_batch(symbols):
    tasks = [predict_async(s) for s in symbols]
    return await asyncio.gather(*tasks)
```

---

**📌 Salve esta referência para consultas rápidas!**

*Última atualização: Dezembro 2024*

