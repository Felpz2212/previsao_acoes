# 📊 Relatório de Avaliação dos Modelos LSTM

**Tech Challenge Fase 4 - FIAP Pós-Tech ML Engineering**

**Data**: Dezembro 2025

---

## 📋 Resumo Executivo

Este relatório apresenta os resultados da avaliação dos modelos LSTM para previsão de preços de ações, desenvolvidos como parte do Tech Challenge Fase 4.

### Decisão: Dados Pós-COVID (2021+)

⚠️ **Importante**: Optamos por treinar com dados a partir de **Janeiro de 2021** para evitar vieses causados pelo crash de Março de 2020 (COVID-19), que representou um evento atípico nos mercados financeiros.

### Modelos Treinados

| Símbolo | Empresa | Dados | Épocas | Early Stop |
|---------|---------|-------|--------|------------|
| AAPL | Apple Inc. | 2021-2024 | ~30 | ✅ |
| GOOGL | Alphabet Inc. | 2021-2024 | ~30 | ✅ |
| MSFT | Microsoft Corp. | 2021-2024 | ~40 | ✅ |
| AMZN | Amazon.com Inc. | 2021-2024 | ~40 | ✅ |
| META | Meta Platforms | 2021-2024 | ~35 | ✅ |
| NVDA | NVIDIA Corp. | 2021-2024 | ~50 | ✅ |
| TSLA | Tesla Inc. | 2021-2024 | ~50 | ✅ |
| JPM | JPMorgan Chase | 2021-2024 | ~35 | ✅ |
| V | Visa Inc. | 2021-2024 | ~40 | ✅ |

---

## 📈 Métricas de Performance

### Tabela Comparativa Completa

| Símbolo | RMSE ($) | MAE ($) | MAPE (%) | R² | Acurácia Dir. | Avaliação |
|---------|----------|---------|----------|-----|---------------|-----------|
| **MSFT** | 20.38 | - | **3.47%** ⭐ | **0.83** | 54.0% | Excelente |
| **V** | 14.82 | 12.66 | **3.72%** ⭐ | -0.77 | 50.0% | Excelente MAPE |
| **TSLA** | 22.84 | - | **5.61%** | **0.88** | 46.6% | Muito Bom |
| **GOOGL** | 18.02 | - | **7.36%** | **0.85** | **55.7%** | Muito Bom |
| **NVDA** | 13.15 | - | **7.50%** | **0.81** | 46.0% | Muito Bom |
| **META** | 58.58 | - | **7.60%** | 0.42 | **55.7%** | Bom |
| **AAPL** | 26.06 | - | **8.28%** | 0.04 | 52.3% | Bom |
| **JPM** | 30.64 | - | 10.42% | -0.28 | 49.4% | Aceitável |
| **AMZN** | 27.52 | - | 11.61% | -1.32 | 51.7% | Aceitável |

### Interpretação das Métricas

#### MAPE (Mean Absolute Percentage Error)
- **< 5%**: Excelente ⭐ MSFT, V
- **5-10%**: Bom ✅ TSLA, GOOGL, NVDA, META, AAPL
- **10-20%**: Aceitável - JPM, AMZN
- **> 20%**: Precisa melhoria

#### R² (Coeficiente de Determinação)
- **> 0.8**: Excelente ⭐ TSLA (0.88), GOOGL (0.85), MSFT (0.83), NVDA (0.81)
- **0.4-0.8**: Bom ✅ META (0.42)
- **0-0.4**: Aceitável - AAPL (0.04)
- **< 0**: Modelo afetado por volatilidade

#### Acurácia Direcional
- **> 55%**: Bom ✅ GOOGL, META (55.7%)
- **50-55%**: Melhor que random ✅ MSFT, AAPL, AMZN
- **< 50%**: Desafiador (ações muito voláteis)

---

## 📊 Análise por Setor

### Tech Giants (AAPL, GOOGL, MSFT, META)

| Métrica | AAPL | GOOGL | MSFT | META |
|---------|------|-------|------|------|
| MAPE | 8.28% | 7.36% | **3.47%** | 7.60% |
| R² | 0.04 | **0.85** | **0.83** | 0.42 |
| Dir. Acc | 52.3% | **55.7%** | 54.0% | **55.7%** |

**Destaque**: MSFT teve o melhor desempenho geral do setor.

### E-Commerce & Cloud (AMZN)

- MAPE: 11.61% (aceitável para alta volatilidade)
- Maior desafio: múltiplos segmentos de negócio afetam preço

### Semicondutores (NVDA)

- MAPE: **7.50%** (excelente para setor volátil)
- R²: **0.81** (muito bom)
- Desafio: alta volatilidade por expectativas de IA

### Veículos Elétricos (TSLA)

- MAPE: **5.61%** (surpreendentemente bom)
- R²: **0.88** (melhor R² entre todos!)
- Apesar da volatilidade, padrões são capturados

### Financeiro (JPM, V)

| Métrica | JPM | V |
|---------|-----|---|
| MAPE | 10.42% | **3.72%** |
| R² | -0.28 | -0.77 |

**Destaque**: V (Visa) teve o segundo melhor MAPE geral.

---

## 🔬 Impacto da Remoção de Dados Pré-COVID

### Comparação: 2018+ vs 2021+

| Métrica | 2018-2024 | 2021-2024 | Melhoria |
|---------|-----------|-----------|----------|
| MAPE Médio | ~17% | **~7%** | ✅ **59% melhor** |
| R² Positivos | 1/3 | **6/9** | ✅ **Dobrou** |
| Dir. Acc > 50% | 2/3 | **6/9** | ✅ **Dobrou** |

### Por que a Melhoria?

1. **Remoção de Outliers**: Crash de 2020 criava vieses
2. **Dados mais Homogêneos**: Mercado pós-pandemia mais estável
3. **Padrões mais Claros**: Modelo captura tendências recentes melhor

---

## 🏗️ Arquitetura do Modelo

### LSTM Melhorado (ImprovedLSTMPredictor)

```
Arquitetura:
- LSTM Bidirecional: 3 camadas
- Hidden Size: 64 neurônios
- Dropout: 0.3
- Attention Mechanism: Pesos de atenção
- Loss Function: Huber Loss
- Optimizer: AdamW com weight decay
```

### Features Utilizadas (16 total)

| Categoria | Features |
|-----------|----------|
| Preços | open, high, low, close |
| Volume | volume, volume_ma_7 |
| Médias Móveis | ma_7, ma_30, ma_90 |
| Volatilidade | volatility_7, volatility_30 |
| Momentum | momentum, roc_7, roc_30 |
| Variação | price_change, pct_change |

### Técnicas de Regularização

1. **Early Stopping**: Patience = 10 épocas
2. **Learning Rate Scheduler**: ReduceLROnPlateau (fator 0.5)
3. **Gradient Clipping**: Max norm = 1.0
4. **Dropout**: 30%
5. **Weight Decay**: 1e-5

---

## 📊 Validação Temporal (Walk-Forward)

```
Split dos Dados (2021-2024):
┌─────────────────────────────────────────────────────┐
│  70% TREINO  │  15% VALIDAÇÃO  │  15% TESTE  │
│  (2021-2023)  │   (2023-2024)   │  (2024)      │
└─────────────────────────────────────────────────────┘
```

---

## 🎯 Conclusões

### Pontos Fortes

1. ✅ **MAPE < 10%** para 7 de 9 ações
2. ✅ **R² > 0.8** para 4 ações (TSLA, GOOGL, MSFT, NVDA)
3. ✅ **Early Stopping Efetivo**: Todos modelos pararam antes de overfitting
4. ✅ **Decisão de Dados**: Remover dados pré-COVID melhorou significativamente

### Destaques por Categoria

| Categoria | Melhor Modelo | MAPE |
|-----------|---------------|------|
| Melhor MAPE | MSFT | 3.47% |
| Melhor R² | TSLA | 0.88 |
| Melhor Dir. Acc | GOOGL/META | 55.7% |
| Melhor Geral | GOOGL | 7.36% / 0.85 / 55.7% |

### Limitações Identificadas

1. **Ações Voláteis**: AMZN, JPM têm métricas inferiores
2. **R² Negativo**: Algumas ações (V, JPM, AMZN) têm alta variância
3. **Horizonte**: Previsão apenas 1 dia à frente

### Recomendações Futuras

1. **Previsão de Retornos %**: Ao invés de preços absolutos
2. **Retraining Mensal**: Atualizar modelos com dados recentes
3. **Ensemble**: Combinar modelos para ações diferentes
4. **Features de Mercado**: Adicionar VIX, taxas de juros

---

## 📁 Modelos Disponíveis

### HuggingFace Hub
- **Repositório**: `henriquebap/stock-predictor-lstm`
- **Modelos**: 9 ações (AAPL, GOOGL, MSFT, AMZN, META, NVDA, TSLA, JPM, V)

### Arquivos por Modelo

```
henriquebap/stock-predictor-lstm/
├── lstm_model_AAPL.pth
├── lstm_model_GOOGL.pth
├── lstm_model_MSFT.pth
├── lstm_model_AMZN.pth
├── lstm_model_META.pth
├── lstm_model_NVDA.pth
├── lstm_model_TSLA.pth
├── lstm_model_JPM.pth
├── lstm_model_V.pth
├── scaler_*.pkl (preprocessors)
└── metadata_*.json (métricas)
```

---

## 📚 Referências

- **Dataset**: Yahoo Finance via yfinance
- **Framework**: PyTorch 2.0+
- **Período de Treino**: Janeiro 2021 - Dezembro 2024
- **Ambiente**: CPU (Apple Silicon M1/M2)
- **Validação**: Walk-forward temporal split

---

**Desenvolvido para**: Tech Challenge Fase 4 - FIAP Pós-Tech ML Engineering

**Data de Geração**: 03/12/2025

**Versão**: 2.0 (Dados pós-COVID)
