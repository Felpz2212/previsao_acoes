"""
Stock Price Predictor - HuggingFace Spaces
LSTM-based stock price prediction.

Tech Challenge Fase 4 - FIAP Pós-Tech ML Engineering
"""
import gradio as gr
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LSTM MODEL
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=16, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
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
# DATA FUNCTIONS
# ============================================================================

def load_data(symbol, days=400):
    end = datetime.now()
    start = end - timedelta(days=days)
    
    df = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
    
    if df.empty:
        raise ValueError(f"Sem dados para {symbol}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = df.columns.str.lower()
    df = df.reset_index()
    
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'timestamp'})
    
    return df


def create_features(df):
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


def prepare_sequences(df, seq_len=60):
    cols = ['open', 'high', 'low', 'close', 'volume',
            'price_change', 'high_low_pct', 'close_open_pct',
            'ma_7', 'ma_30', 'ma_90', 'volatility_7', 'volatility_30',
            'volume_change', 'volume_ma_7', 'momentum']
    
    df = create_features(df)
    
    features = df[cols].values
    target = df['close'].values
    
    f_scaler = MinMaxScaler()
    features_scaled = f_scaler.fit_transform(features)
    
    t_scaler = MinMaxScaler()
    target_scaled = t_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    X, y = [], []
    for i in range(seq_len, len(features_scaled)):
        X.append(features_scaled[i-seq_len:i])
        y.append(target_scaled[i])
    
    return np.array(X), np.array(y), f_scaler, t_scaler, df


def train_model(X, y, epochs=25):
    model = LSTMModel(input_size=X.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    batch = 32
    for _ in range(epochs):
        idx = np.random.permutation(len(X))
        for i in range(0, len(X), batch):
            b_idx = idx[i:i+batch]
            xb = torch.FloatTensor(X[b_idx])
            yb = torch.FloatTensor(y[b_idx]).reshape(-1, 1)
            
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    
    return model


# ============================================================================
# CACHE & PREDICTION
# ============================================================================

cache = {}


def predict(symbol):
    symbol = symbol.upper().strip()
    
    if not symbol:
        return "❌ Digite um símbolo (ex: AAPL, GOOGL)"
    
    try:
        df = load_data(symbol)
        
        if len(df) < 120:
            return f"❌ Dados insuficientes para {symbol}"
        
        current = float(df['close'].iloc[-1])
        
        # Get or train model
        if symbol not in cache:
            X, y, f_sc, t_sc, _ = prepare_sequences(df)
            split = int(0.8 * len(X))
            model = train_model(X[:split], y[:split])
            cache[symbol] = {'model': model, 'f_sc': f_sc, 't_sc': t_sc}
        
        data = cache[symbol]
        
        # Prepare prediction input
        df_p = create_features(df)
        cols = ['open', 'high', 'low', 'close', 'volume',
                'price_change', 'high_low_pct', 'close_open_pct',
                'ma_7', 'ma_30', 'ma_90', 'volatility_7', 'volatility_30',
                'volume_change', 'volume_ma_7', 'momentum']
        
        feat = data['f_sc'].transform(df_p[cols].values)
        x = torch.FloatTensor(feat[-60:].reshape(1, 60, 16))
        
        # Predict
        data['model'].eval()
        with torch.no_grad():
            pred_scaled = data['model'](x).numpy()[0, 0]
        
        predicted = data['t_sc'].inverse_transform([[pred_scaled]])[0, 0]
        change = ((predicted - current) / current) * 100
        
        direction = "📈 ALTA" if change > 0 else "📉 BAIXA"
        
        return f"""
# {direction} prevista para {symbol}

## Resultados

| Métrica | Valor |
|---------|-------|
| **Preço Atual** | ${current:.2f} |
| **Previsão** | ${predicted:.2f} |
| **Variação** | {change:+.2f}% |
| **Data** | {datetime.now().strftime('%d/%m/%Y')} |

---

⚠️ **Aviso**: Esta é uma previsão educacional usando LSTM.  
**NÃO** use para decisões de investimento reais!

---

🧠 *Modelo treinado com {len(df)} dias de dados históricos*
"""
        
    except Exception as e:
        return f"❌ Erro: {str(e)}"


# ============================================================================
# GRADIO APP
# ============================================================================

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Símbolo da Ação",
        placeholder="Ex: AAPL, GOOGL, MSFT, NVDA...",
        value="AAPL"
    ),
    outputs=gr.Markdown(),
    title="📈 Stock Price Predictor - LSTM",
    description="""
    Sistema de previsão de preços de ações usando **Deep Learning (LSTM)**.
    
    🎓 **Tech Challenge Fase 4** - FIAP Pós-Tech ML Engineering
    
    ---
    
    **Ações populares**: AAPL, GOOGL, MSFT, AMZN, NVDA, TSLA, META
    """,
    article="""
    ### 🧠 Como Funciona?
    
    1. **Dados**: Baixa histórico do Yahoo Finance
    2. **Features**: Cria 16 indicadores técnicos
    3. **Modelo**: LSTM com 2 camadas e 50 neurônios
    4. **Previsão**: Usa últimos 60 dias para prever próximo dia
    
    ---
    
    **Tech Challenge Fase 4** | FIAP Pós-Tech | Dezembro 2024
    """,
    examples=[
        ["AAPL"],
        ["GOOGL"],
        ["MSFT"],
        ["NVDA"],
        ["TSLA"]
    ],
    cache_examples=False
)


if __name__ == "__main__":
    demo.launch()
