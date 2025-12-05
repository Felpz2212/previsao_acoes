---
license: mit
tags:
  - pytorch
  - time-series
  - stock-prediction
  - lstm
  - finance
language:
  - pt
  - en
pipeline_tag: time-series-forecasting
---

# Stock Price Predictor - LSTM Models

Modelos LSTM para previsão de preços de ações treinados como parte do **Tech Challenge Fase 4** - FIAP Pós-Tech Machine Learning Engineering.

## 📊 Modelos Disponíveis

| Modelo | Tipo | Métricas |
|--------|------|----------|
| `lstm_model_BASE.pth` | Genérico (5 ações) | Dir. Acc: 79% |
| `lstm_model_AAPL.pth` | Apple | MAPE: 2.8%, R²: 88.9% |
| `lstm_model_GOOGL.pth` | Google | MAPE: 3.9%, R²: 76.4% |
| `lstm_model_NVDA.pth` | NVIDIA (fine-tuned) | - |

## 🧠 Arquitetura

```
LSTM Neural Network
├── Input: 16 features técnicas
├── LSTM Layer 1: 50 hidden units
├── Dropout: 0.2
├── LSTM Layer 2: 50 hidden units
├── Dropout: 0.2
└── Output: 1 (preço previsto)
```

## 📈 Features Utilizadas (16)

1. Open, High, Low, Close, Volume
2. Price Change %, High-Low %, Close-Open %
3. Moving Averages: 7, 30, 90 dias
4. Volatility: 7, 30 dias
5. Volume Change, Volume MA 7
6. Momentum

## 🔧 Como Usar

```python
import torch
from huggingface_hub import hf_hub_download

# Baixar modelo
model_path = hf_hub_download(
    repo_id="henriquebap/stock-predictor-lstm",
    filename="lstm_model_AAPL.pth"
)

# Carregar
checkpoint = torch.load(model_path, map_location='cpu')
```

## 📁 Arquivos

- `lstm_model_*.pth` - Modelos PyTorch
- `scaler_*.pkl` - Preprocessadores (MinMaxScaler)
- `metadata_*.json` - Metadados de treinamento

## ⚠️ Disclaimer

Projeto educacional. NÃO use para decisões de investimento reais!

## 👨‍💻 Desenvolvido por

Tech Challenge Fase 4 Team - FIAP Pós-Tech MLET | Dezembro 2024

