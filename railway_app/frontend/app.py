"""
Stock Predictor - Frontend Streamlit
Tech Challenge Fase 4 - FIAP Pós-Tech ML Engineering
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os

from components.charts import create_candlestick_chart, create_comparison_chart
from components.sidebar import render_sidebar
from components.predictions import render_prediction_card

# Configuração - Garante que URL tem schema https://
_api_url = os.getenv("API_URL", "http://localhost:8000")
if _api_url and not _api_url.startswith(("http://", "https://")):
    _api_url = f"https://{_api_url}"
API_URL = _api_url.rstrip("/")

# Config da página
st.set_page_config(
    page_title="Stock Predictor LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #333;
    }
    .prediction-up {
        color: #00ff88;
        font-weight: bold;
    }
    .prediction-down {
        color: #ff4757;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


def get_stock_data(symbol: str, days: int = 365) -> dict:
    """Obtém dados da API."""
    try:
        response = requests.get(
            f"{API_URL}/api/stocks/{symbol}",
            params={"days": days},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Erro de conexão: {e}")
        return None


def get_prediction(symbol: str) -> dict:
    """Obtém previsão da API."""
    try:
        response = requests.get(
            f"{API_URL}/api/predictions/{symbol}",
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Erro ao obter previsão: {e}")
        return None


def get_popular_stocks() -> dict:
    """Obtém lista de ações populares."""
    try:
        response = requests.get(f"{API_URL}/api/stocks/popular/list", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback local
    return {
        "categories": {
            "🇺🇸 Tech US": ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"],
            "💰 Finance US": ["JPM", "BAC", "V", "MA"],
            "🇧🇷 Brasil B3": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
        }
    }


def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">📈 Stock Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Previsão de preços com Deep Learning (LSTM)</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p style='text-align: right; color: #888;'>🕐 {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    selected_symbol, selected_days, compare_mode, compare_symbols = render_sidebar()
    
    # Main content
    if compare_mode and compare_symbols:
        # Modo comparação
        st.subheader(f"📊 Comparação: {', '.join(compare_symbols)}")
        
        all_data = {}
        for sym in compare_symbols:
            data = get_stock_data(sym, selected_days)
            if data:
                all_data[sym] = data
        
        if all_data:
            fig = create_comparison_chart(all_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de performance
            st.subheader("📈 Performance")
            perf_data = []
            for sym, data in all_data.items():
                df = pd.DataFrame(data['data'])
                start_price = df['close'].iloc[0]
                end_price = df['close'].iloc[-1]
                perf = ((end_price - start_price) / start_price) * 100
                perf_data.append({
                    "Símbolo": sym,
                    "Nome": data['name'],
                    "Preço Atual": f"${end_price:.2f}",
                    "Performance": f"{perf:+.2f}%"
                })
            
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
    else:
        # Modo normal - uma ação
        if selected_symbol:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📊 {selected_symbol}")
                
                # Obter dados
                with st.spinner("Carregando dados..."):
                    stock_data = get_stock_data(selected_symbol, selected_days)
                
                if stock_data:
                    # Gráfico
                    df = pd.DataFrame(stock_data['data'])
                    fig = create_candlestick_chart(df, selected_symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Indicadores
                    indicators = stock_data.get('indicators', {})
                    ind_cols = st.columns(4)
                    
                    with ind_cols[0]:
                        st.metric("MA 7", f"${indicators.get('ma_7', 0):.2f}")
                    with ind_cols[1]:
                        st.metric("MA 30", f"${indicators.get('ma_30', 0):.2f}")
                    with ind_cols[2]:
                        st.metric("Volatilidade", f"${indicators.get('volatility', 0):.2f}")
                    with ind_cols[3]:
                        trend = indicators.get('trend', 'up')
                        st.metric("Tendência", "📈 Alta" if trend == 'up' else "📉 Baixa")
                else:
                    st.error(f"Não foi possível obter dados para {selected_symbol}")
            
            with col2:
                st.subheader("🔮 Previsão LSTM")
                
                if st.button("🚀 Fazer Previsão", use_container_width=True):
                    with st.spinner("Calculando previsão..."):
                        prediction = get_prediction(selected_symbol)
                    
                    if prediction:
                        render_prediction_card(prediction)
                    else:
                        st.error("Erro ao obter previsão")
                
                # Info do modelo
                st.markdown("---")
                st.markdown("### 🧠 Sobre o Modelo")
                st.markdown("""
                - **Arquitetura**: LSTM 2 camadas
                - **Features**: 16 indicadores técnicos
                - **Período**: 60 dias de histórico
                - **Hub**: [henriquebap/stock-predictor-lstm](https://huggingface.co/henriquebap/stock-predictor-lstm)
                """)
        else:
            # Página inicial
            st.info("👈 Selecione uma ação na barra lateral para começar")
            
            # Mostrar ações populares
            popular = get_popular_stocks()
            
            st.subheader("📋 Ações Populares")
            
            for category, symbols in popular.get('categories', {}).items():
                st.markdown(f"**{category}**")
                cols = st.columns(len(symbols))
                for i, sym in enumerate(symbols):
                    with cols[i]:
                        if st.button(sym, key=f"pop_{sym}"):
                            st.session_state['selected_symbol'] = sym
                            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>⚠️ <strong>Disclaimer</strong>: Previsões educacionais. NÃO use para investimentos reais!</p>
        <p>🎓 Tech Challenge Fase 4 - FIAP Pós-Tech ML Engineering | Dezembro 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

