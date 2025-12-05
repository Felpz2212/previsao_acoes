# 🚀 Guia Rápido de Início

Este guia vai te ajudar a começar rapidamente com o projeto de previsão de ações.

## ⚡ Início Rápido (5 minutos)

### 1. Clone e Instale

```bash
# Clone o repositório
git clone https://github.com/your-username/previsao_acoes.git
cd previsao_acoes

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt
```

### 2. Treine seu Primeiro Modelo

```bash
# Treine um modelo para Apple (AAPL)
python scripts/train_model.py AAPL --start-date 2020-01-01 --end-date 2024-12-31

# Isso vai:
# - Baixar dados históricos do Yahoo Finance
# - Processar e criar features
# - Treinar o modelo LSTM
# - Avaliar com métricas (RMSE, MAE, MAPE)
# - Salvar modelo e preprocessador
```

### 3. Inicie a API

```bash
# Inicie o servidor FastAPI
python -m uvicorn src.api.main:app --reload

# API disponível em: http://localhost:8000
# Documentação: http://localhost:8000/docs
```

### 4. Faça sua Primeira Previsão

```bash
# Via cURL
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 1}'

# Ou acesse http://localhost:8000/docs para usar a interface Swagger
```

## 🎯 Próximos Passos

### Treinar Mais Modelos

```bash
# Google
python scripts/train_model.py GOOGL

# Microsoft
python scripts/train_model.py MSFT

# Tesla
python scripts/train_model.py TSLA
```

### Rodar Testes

```bash
# Todos os testes
pytest

# Apenas testes de API
pytest tests/test_api.py -v
```

### Usar Docker

```bash
# Build e run
docker-compose up --build

# API em http://localhost:8000
```

### Deploy

#### Railway (Backend API)

```bash
# Instale Railway CLI
npm i -g @railway/cli

# Login e deploy
railway login
railway init
railway up
```

#### HuggingFace Spaces (Interface UI)

1. Crie um Space em https://huggingface.co/spaces
2. Clone: `git clone https://huggingface.co/spaces/your-user/stock-prediction`
3. Copie `app_gradio.py` como `app.py`
4. Push: `git add . && git commit -m "Add UI" && git push`

## 📚 Documentação Completa

- **README.md** - Documentação completa do projeto
- **DEPLOYMENT.md** - Guias detalhados de deploy
- **API Docs** - http://localhost:8000/docs (quando rodando)

## 🆘 Problemas Comuns

### Erro: "Model not found"

**Solução:** Treine o modelo primeiro
```bash
python scripts/train_model.py AAPL
```

### Erro: "Port already in use"

**Solução:** Use outra porta
```bash
uvicorn src.api.main:app --port 8001
```

### Erro: "Module not found"

**Solução:** Ative o ambiente virtual
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## 🎓 Estrutura Básica

```
previsao_acoes/
├── src/
│   ├── api/              # FastAPI app
│   ├── data/             # Data loading & preprocessing
│   ├── models/           # LSTM model
│   └── training/         # Training pipeline
├── scripts/              # Utility scripts
├── tests/                # Test suite
├── models/               # Saved models (gerado)
└── data/                 # Data storage (gerado)
```

## 📊 Métricas do Modelo

Após treinar, você verá métricas como:

- **RMSE**: Erro quadrático médio
- **MAE**: Erro absoluto médio
- **MAPE**: Erro percentual absoluto médio
- **R²**: Coeficiente de determinação
- **Directional Accuracy**: Acurácia da direção

Valores bons:
- MAPE < 10%: Excelente
- MAPE 10-20%: Bom
- MAPE 20-50%: Aceitável
- MAPE > 50%: Precisa melhorar

## 🔧 Personalização

### Ajustar Hiperparâmetros

Edite `config/settings.py`:

```python
LSTM_SEQUENCE_LENGTH = 60  # Janela de histórico
LSTM_EPOCHS = 50           # Épocas de treinamento
LSTM_BATCH_SIZE = 32       # Tamanho do batch
LSTM_LEARNING_RATE = 0.001 # Taxa de aprendizado
LSTM_HIDDEN_SIZE = 50      # Neurônios LSTM
LSTM_NUM_LAYERS = 2        # Camadas LSTM
```

### Treinar com Configurações Customizadas

```bash
python scripts/train_model.py AAPL --epochs 100 --batch-size 64
```

## 🎉 Pronto!

Você agora tem:
- ✅ Modelo LSTM treinado
- ✅ API rodando localmente
- ✅ Documentação interativa
- ✅ Testes configurados

Para mais detalhes, consulte o **README.md** completo.

---

**Dúvidas?** Abra uma issue no GitHub!

