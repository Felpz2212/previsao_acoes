"""
Stock Price Predictor - LSTM Real Model
HuggingFace Spaces - FIAP Tech Challenge Fase 4

Usa modelos LSTM treinados do HuggingFace Model Hub.
"""
import gradio as gr
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LSTM MODEL ARCHITECTURE
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM para previsão de séries temporais."""
    
    def __init__(self, input_size=16, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)


# ============================================================================
# MODEL LOADING FROM HUB
# ============================================================================

MODEL_REPO = "henriquebap/stock-predictor-lstm"
AVAILABLE_MODELS = ['AAPL', 'GOOGL', 'BASE']

model_cache = {}


def load_model_from_hub(symbol: str):
    """Carrega modelo do HuggingFace Hub."""
    symbol = symbol.upper()
    
    if symbol in model_cache:
        return model_cache[symbol]
    
    # Verificar se existe modelo específico
    model_file = f"lstm_model_{symbol}.pth"
    scaler_file = f"scaler_{symbol}.pkl"
    
    try:
        # Tentar baixar modelo específico
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=model_file)
        scaler_path = hf_hub_download(repo_id=MODEL_REPO, filename=scaler_file)
        model_type = "específico"
    except:
        # Usar modelo BASE
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename="lstm_model_BASE.pth")
        scaler_path = hf_hub_download(repo_id=MODEL_REPO, filename="scaler_BASE.pkl")
        model_type = "base"
    
    # Carregar modelo
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = LSTMModel(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint.get('dropout', 0.2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Carregar scaler
    scaler_data = joblib.load(scaler_path)
    
    model_cache[symbol] = {
        'model': model,
        'scaler': scaler_data['scaler'],
        'target_scaler': scaler_data['target_scaler'],
        'feature_columns': scaler_data['feature_columns'],
        'type': model_type
    }
    
    return model_cache[symbol]


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def load_stock_data(symbol: str, days: int = 400) -> pd.DataFrame:
    """Carrega dados do Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=days)
    
    df = yf.download(
        symbol,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        progress=False
    )
    
    if df.empty:
        raise ValueError(f"Dados não encontrados para {symbol}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = df.columns.str.lower()
    df = df.reset_index()
    
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'timestamp'})
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features técnicas."""
    df = df.copy()
    
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['high_low_pct'] = ((df['high'] - df['low']) / df['low']).fillna(0)
    df['close_open_pct'] = ((df['close'] - df['open']) / df['open']).fillna(0)
    
    df['ma_7'] = df['close'].rolling(7, min_periods=1).mean()
    df['ma_30'] = df['close'].rolling(30, min_periods=1).mean()
    df['ma_90'] = df['close'].rolling(90, min_periods=1).mean()
    
    df['volatility_7'] = df['close'].rolling(7, min_periods=1).std().fillna(0)
    df['volatility_30'] = df['close'].rolling(30, min_periods=1).std().fillna(0)
    
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    df['volume_ma_7'] = df['volume'].rolling(7, min_periods=1).mean()
    
    df['momentum'] = (df['close'] - df['close'].shift(4)).fillna(0)
    
    return df.bfill().ffill()


# ============================================================================
# PREDICTION WITH LSTM
# ============================================================================

def predict_with_lstm(symbol: str) -> str:
    """Faz previsão usando modelo LSTM do Hub."""
    symbol = symbol.upper().strip()
    
    if not symbol:
        return "❌ Digite um símbolo válido (ex: AAPL, GOOGL, MSFT)"
    
    try:
        # Carregar dados
        df = load_stock_data(symbol)
        
        if len(df) < 70:
            return f"❌ Dados insuficientes para {symbol} (mínimo 70 dias)"
        
        current_price = float(df['close'].iloc[-1])
        
        # Tentar carregar modelo do Hub
        try:
            model_data = load_model_from_hub(symbol)
            model = model_data['model']
            scaler = model_data['scaler']
            target_scaler = model_data['target_scaler']
            feature_cols = model_data['feature_columns']
            model_type = model_data['type']
            using_lstm = True
        except Exception as e:
            # Fallback para modelo simples
            using_lstm = False
            model_type = "fallback"
        
        # Preparar features
        df_feat = create_features(df)
        
        if using_lstm:
            # Usar modelo LSTM
            feature_cols_available = [c for c in feature_cols if c in df_feat.columns]
            
            if len(feature_cols_available) < len(feature_cols):
                # Preencher features faltantes
                for col in feature_cols:
                    if col not in df_feat.columns:
                        df_feat[col] = 0
            
            features = df_feat[feature_cols].values
            features_scaled = scaler.transform(features)
            
            # Preparar sequência (últimos 60 dias)
            seq_len = 60
            X = features_scaled[-seq_len:].reshape(1, seq_len, len(feature_cols))
            X_tensor = torch.FloatTensor(X)
            
            # Previsão
            with torch.no_grad():
                pred_scaled = model(X_tensor).numpy()[0, 0]
            
            predicted_price = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
        else:
            # Fallback: média móvel ponderada
            ma_7 = float(df_feat['ma_7'].iloc[-1])
            ma_30 = float(df_feat['ma_30'].iloc[-1])
            momentum = float(df_feat['momentum'].iloc[-1])
            
            predicted_price = current_price + (momentum * 0.3) + ((ma_7 - ma_30) * 0.2)
        
        # Calcular métricas
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        if change_pct > 1:
            direction = "📈 ALTA"
            emoji = "🟢"
        elif change_pct < -1:
            direction = "📉 BAIXA"
            emoji = "🔴"
        else:
            direction = "➡️ ESTÁVEL"
            emoji = "🟡"
        
        # Indicadores técnicos
        ma_7 = float(df_feat['ma_7'].iloc[-1])
        ma_30 = float(df_feat['ma_30'].iloc[-1])
        volatility = float(df_feat['volatility_7'].iloc[-1])
        
        trend = "📈 Positiva" if ma_7 > ma_30 else "📉 Negativa"
        
        # Performance recente
        week_ago = float(df['close'].iloc[-5]) if len(df) > 5 else current_price
        month_ago = float(df['close'].iloc[-21]) if len(df) > 21 else current_price
        week_change = ((current_price - week_ago) / week_ago) * 100
        month_change = ((current_price - month_ago) / month_ago) * 100
        
        result = f"""
# {emoji} {direction} prevista para {symbol}

## 🤖 Modelo: LSTM {"Específico" if model_type == "específico" else "Base" if model_type == "base" else "Fallback"}

| Métrica | Valor |
|---------|-------|
| **Preço Atual** | ${current_price:.2f} |
| **Previsão LSTM** | ${predicted_price:.2f} |
| **Variação Esperada** | {change_pct:+.2f}% |

---

## 📊 Indicadores Técnicos

| Indicador | Valor |
|-----------|-------|
| **MA 7 dias** | ${ma_7:.2f} |
| **MA 30 dias** | ${ma_30:.2f} |
| **Tendência** | {trend} |
| **Volatilidade** | ${volatility:.2f} |

---

## 📅 Performance

| Período | Variação |
|---------|----------|
| **Semana** | {week_change:+.2f}% |
| **Mês** | {month_change:+.2f}% |

---

📦 **Modelo**: [henriquebap/stock-predictor-lstm](https://huggingface.co/henriquebap/stock-predictor-lstm)

⚠️ **Disclaimer**: Previsão educacional. NÃO use para investimentos reais!

*Tech Challenge Fase 4 - FIAP Pós-Tech MLET*
"""
        return result
        
    except Exception as e:
        return f"❌ Erro: {str(e)}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

demo = gr.Interface(
    fn=predict_with_lstm,
    inputs=gr.Textbox(
        label="Símbolo da Ação",
        placeholder="Ex: AAPL, GOOGL, MSFT, AMZN",
        value="AAPL"
    ),
    outputs=gr.Markdown(label="Previsão LSTM"),
    title="📈 Stock Price Predictor - LSTM",
    description="""
    ### Sistema de Previsão com Deep Learning (LSTM)
    
    🎓 **Tech Challenge Fase 4** - FIAP Pós-Tech Machine Learning Engineering
    
    Usa modelos LSTM treinados disponíveis no [HuggingFace Hub](https://huggingface.co/henriquebap/stock-predictor-lstm).
    
    **Modelos treinados**: AAPL, GOOGL (outros usam modelo BASE)
    """,
    article="""
    ### 🧠 Arquitetura LSTM
    
    - **Input**: 16 features técnicas (60 dias)
    - **LSTM**: 2 camadas × 50 neurônios
    - **Output**: Preço previsto
    
    ### 📊 Features
    
    Preços, Médias Móveis, Volatilidade, Momentum, Volume
    
    ---
    
    **GitHub**: [previsao_acoes](https://github.com/henriquebap/previsao_acoes)
    
    **Dezembro 2024** | FIAP Pós-Tech MLET
    """,
    examples=[
        ["AAPL"],
        ["GOOGL"],
        ["MSFT"],
        ["AMZN"],
        ["TSLA"]
    ],
    cache_examples=False
)


if __name__ == "__main__":
    demo.launch()
