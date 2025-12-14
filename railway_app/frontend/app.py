"""
Stock Predictor - Frontend Streamlit
Tech Challenge Fase 4 - FIAP Pós-Tech ML Engineering
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

# CSS customizado - UI MODERNA
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        text-align: center;
        animation: gradient 3s ease infinite;
    }
    
    .sub-header {
        color: #aaa;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        text-align: center;
        font-weight: 300;
    }
    
    /* Cards de métrica */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Botões estilizados */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 0.75rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Botão primário especial */
    button[kind="primary"] {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    }
    
    /* Cards de previsão */
    .prediction-up {
        color: #00ff88;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .prediction-down {
        color: #ff4757;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 71, 87, 0.5);
    }
    
    /* Loading spinner customizado */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Expander estilizado */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 0.75rem;
        border-left: 4px solid #667eea;
    }
    
    /* Sidebar melhorada */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Animações suaves */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    /* Scrollbar customizada */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
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


def render_monitoring_page():
    """Página de Monitoramento."""
    st.markdown('<h1 class="main-header">📊 Monitoramento</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Métricas da API e Modelos em tempo real</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Tabs principais
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🧠 Modelos", "📝 Prometheus"])
    
    # Buscar dados
    try:
        response = requests.get(f"{API_URL}/api/monitoring", timeout=10)
        data = response.json() if response.status_code == 200 else None
    except:
        data = None
    
    with tab1:
        col_refresh, col_status = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Atualizar", key="refresh_overview"):
                st.rerun()
        with col_status:
            if data:
                st.success(f"✅ API Online | Última atualização: {datetime.now().strftime('%H:%M:%S')}")
            else:
                st.error("❌ API Offline")
        
        if data:
            # KPIs em cards coloridos
            st.markdown("### 📊 Indicadores Principais")
            cols = st.columns(5)
            
            with cols[0]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="color: white; margin: 0;">⏱️ Uptime</h3>
                    <h2 style="color: white; margin: 0.5rem 0;">{data.get('uptime_human', 'N/A')}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="color: white; margin: 0;">📨 Requests</h3>
                    <h2 style="color: white; margin: 0.5rem 0;">{data.get('total_requests', 0):,}</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem;">{data.get('requests_per_minute', 0):.1f}/min</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                error_rate = data.get('error_rate_percent', 0)
                error_color = "#11998e, #38ef7d" if error_rate < 5 else "#ff416c, #ff4b2b"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {error_color}); 
                            padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="color: white; margin: 0;">❌ Erros</h3>
                    <h2 style="color: white; margin: 0.5rem 0;">{error_rate:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="color: white; margin: 0;">🔮 Previsões</h3>
                    <h2 style="color: white; margin: 0.5rem 0;">{data.get('total_predictions', 0):,}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[4]:
                system = data.get('system', {}) or {}
                cpu = system.get('cpu_percent', 0)
                cpu_color = "#11998e, #38ef7d" if cpu < 70 else "#ff416c, #ff4b2b"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {cpu_color}); 
                            padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="color: white; margin: 0;">💻 CPU</h3>
                    <h2 style="color: white; margin: 0.5rem 0;">{cpu:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gráfico de Latência por Endpoint
            st.markdown("### ⚡ Latência por Endpoint")
            endpoints = data.get('endpoints', {})
            if endpoints:
                endpoint_data = []
                for ep, stats in endpoints.items():
                    endpoint_data.append({
                        'Endpoint': ep,
                        'Requests': stats.get('count', 0),
                        'Avg (ms)': round(stats.get('avg_time_ms', 0), 1),
                        'Max (ms)': round(stats.get('max_time_ms', 0), 1),
                        'Errors %': round(stats.get('error_rate', 0), 1)
                    })
                
                df = pd.DataFrame(endpoint_data).sort_values('Requests', ascending=False)
                
                # Gráfico de barras
                fig = px.bar(
                    df.head(8),
                    x='Endpoint',
                    y='Avg (ms)',
                    color='Avg (ms)',
                    color_continuous_scale='RdYlGn_r',
                    title='Tempo Médio de Resposta'
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("📊 Faça algumas requisições para ver as métricas")
            
            # Sistema
            st.markdown("### 💻 Recursos do Sistema")
            system = data.get('system')
            if system:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cpu_val = system.get('cpu_percent', 0)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=cpu_val,
                        title={'text': "CPU"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, 50], 'color': "#e8f5e9"},
                                {'range': [50, 80], 'color': "#fff9c4"},
                                {'range': [80, 100], 'color': "#ffcdd2"}
                            ]
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    mem_val = system.get('memory_percent', 0)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=mem_val,
                        title={'text': "Memória"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#764ba2"},
                            'steps': [
                                {'range': [0, 60], 'color': "#e8f5e9"},
                                {'range': [60, 85], 'color': "#fff9c4"},
                                {'range': [85, 100], 'color': "#ffcdd2"}
                            ]
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 1rem; border-radius: 0.5rem; height: 180px;">
                        <h4 style="color: #888; margin: 0;">💾 Memória Usada</h4>
                        <h2 style="color: white; margin: 0.5rem 0;">{system.get('memory_used_mb', 0)/1024:.1f} GB</h2>
                        <h4 style="color: #888; margin-top: 1rem;">📊 Disponível</h4>
                        <h2 style="color: #38ef7d; margin: 0.5rem 0;">{system.get('memory_available_mb', 0)/1024:.1f} GB</h2>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("⚠️ Não foi possível conectar ao serviço de monitoramento")
            st.code(f"API URL: {API_URL}/api/monitoring")
    
    with tab2:
        st.markdown("### 🧠 Performance dos Modelos LSTM")
        
        if data:
            models = data.get('models', {})
            
            if models:
                model_data = []
                for sym, stats in models.items():
                    model_data.append({
                        'Símbolo': sym,
                        'Previsões': stats.get('predictions', 0),
                        'Tempo Médio (ms)': round(stats.get('avg_inference_ms', 0), 1),
                        'Mín (ms)': round(stats.get('min_inference_ms', 0), 1),
                        'Máx (ms)': round(stats.get('max_inference_ms', 0), 1),
                    })
                
                df = pd.DataFrame(model_data).sort_values('Previsões', ascending=False)
                
                # Gráfico
                fig = px.bar(
                    df,
                    x='Símbolo',
                    y='Tempo Médio (ms)',
                    color='Previsões',
                    color_continuous_scale='Blues',
                    title='Tempo de Inferência por Modelo'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Nenhuma previsão registrada ainda")
                st.info("👉 **Faça uma previsão na página principal** para ver as métricas do modelo aqui!")
                st.markdown("""
                1. Vá para 🏠 **Principal** no menu lateral
                2. Selecione uma ação (ex: AAPL)
                3. Clique em **🚀 Fazer Previsão**
                4. Volte aqui e clique em **🔄 Atualizar**
                """)
        
        # Métricas de qualidade (estáticas - do treinamento)
        st.markdown("### 📊 Qualidade dos Modelos (Treinamento)")
        
        quality_data = [
            {"Símbolo": "MSFT", "MAPE (%)": 3.47, "R²": 0.83, "Acurácia Dir.": "54.0%", "Status": "🟢 Excelente"},
            {"Símbolo": "V", "MAPE (%)": 3.72, "R²": -0.77, "Acurácia Dir.": "50.0%", "Status": "🟢 Excelente"},
            {"Símbolo": "TSLA", "MAPE (%)": 5.61, "R²": 0.88, "Acurácia Dir.": "46.6%", "Status": "🟡 Bom"},
            {"Símbolo": "GOOGL", "MAPE (%)": 7.36, "R²": 0.85, "Acurácia Dir.": "55.7%", "Status": "🟡 Bom"},
            {"Símbolo": "NVDA", "MAPE (%)": 7.50, "R²": 0.81, "Acurácia Dir.": "46.0%", "Status": "🟡 Bom"},
            {"Símbolo": "META", "MAPE (%)": 7.60, "R²": 0.42, "Acurácia Dir.": "55.7%", "Status": "🟡 Bom"},
            {"Símbolo": "AAPL", "MAPE (%)": 8.28, "R²": 0.04, "Acurácia Dir.": "52.3%", "Status": "🟡 Bom"},
            {"Símbolo": "JPM", "MAPE (%)": 10.42, "R²": -0.28, "Acurácia Dir.": "49.4%", "Status": "🟠 Aceitável"},
            {"Símbolo": "AMZN", "MAPE (%)": 11.61, "R²": -1.32, "Acurácia Dir.": "51.7%", "Status": "🟠 Aceitável"},
        ]
        
        df_quality = pd.DataFrame(quality_data)
        st.dataframe(df_quality, use_container_width=True, hide_index=True)
        
        st.caption("📌 MAPE < 10% é considerado aceitável para previsão de ações")
    
    with tab3:
        st.markdown("### 📝 Métricas Prometheus (Raw)")
        st.markdown("Endpoint: `/metrics` - Formato padrão Prometheus para scraping")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            load_metrics = st.button("🔄 Carregar Métricas", type="primary")
        
        if load_metrics:
            try:
                resp = requests.get(f"{API_URL}/metrics", timeout=10)
                if resp.status_code == 200:
                    metrics_text = resp.text
                    
                    # Estatísticas
                    lines = metrics_text.split('\n')
                    metric_lines = [l for l in lines if l and not l.startswith('#')]
                    help_lines = [l for l in lines if l.startswith('# HELP')]
                    
                    st.success(f"✅ Carregado: {len(metric_lines)} métricas, {len(help_lines)} definições")
                    
                    # Filtro
                    filter_text = st.text_input("🔍 Filtrar métricas", placeholder="Ex: stock_predictor, requests, cpu")
                    
                    # Parse e exibição organizada
                    st.markdown("#### 📊 Métricas Disponíveis")
                    
                    # Agrupar por categoria
                    categories = {
                        "🌐 Requisições HTTP": [],
                        "🔮 Previsões": [],
                        "💻 Sistema": [],
                        "📊 Outras": []
                    }
                    
                    for line in metric_lines:
                        if filter_text and filter_text.lower() not in line.lower():
                            continue
                        
                        if 'request' in line.lower():
                            categories["🌐 Requisições HTTP"].append(line)
                        elif 'prediction' in line.lower() or 'model' in line.lower() or 'inference' in line.lower():
                            categories["🔮 Previsões"].append(line)
                        elif 'cpu' in line.lower() or 'memory' in line.lower() or 'system' in line.lower():
                            categories["💻 Sistema"].append(line)
                        else:
                            categories["📊 Outras"].append(line)
                    
                    for category, metrics in categories.items():
                        if metrics:
                            with st.expander(f"{category} ({len(metrics)} métricas)", expanded=category == "🔮 Previsões"):
                                for m in metrics[:20]:  # Limitar a 20 por categoria
                                    st.code(m, language="text")
                                if len(metrics) > 20:
                                    st.caption(f"... e mais {len(metrics) - 20} métricas")
                    
                    # Raw completo
                    with st.expander("📄 Resposta Raw Completa"):
                        st.code(metrics_text, language="text")
                else:
                    st.error(f"Erro: {resp.status_code}")
            except Exception as e:
                st.error(f"Erro ao carregar: {e}")
        else:
            st.info("👆 Clique no botão acima para carregar as métricas Prometheus")
        
        # Info sobre integração
        st.markdown("---")
        st.markdown("### 🔗 Integração com Grafana Cloud")
        st.markdown(f"""
        Para visualizar no Grafana Cloud, configure o scraping:
        
        ```yaml
        scrape_configs:
          - job_name: 'stock-predictor'
            static_configs:
              - targets: ['previsaoacoes-back-production.up.railway.app']
            scheme: https
            metrics_path: '/metrics'
        ```
        
        **Endpoint direto**: `{API_URL}/metrics`
        """)


def main():
    # Navegação
    page = st.sidebar.radio(
        "📍 Navegação",
        ["🏠 Principal", "📊 Monitoramento"],
        label_visibility="collapsed"
    )
    
    if page == "📊 Monitoramento":
        render_monitoring_page()
        return
    
    # Header aprimorado
    st.markdown('<h1 class="main-header">📈 Stock Predictor LSTM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Previsão de preços com Deep Learning • FIAP Pós-Tech ML Engineering</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 0.5rem; background: rgba(102, 126, 234, 0.1); 
                    border-radius: 0.5rem; margin: 1rem 0;'>
            <span style='color: #888; font-size: 0.9rem;'>🕐 {datetime.now().strftime('%d/%m/%Y • %H:%M:%S')}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar - carregamento automático
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
            # Banner com símbolo selecionado
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem; border-radius: 1rem; margin-bottom: 1rem; text-align: center;'>
                <h2 style='color: white; margin: 0;'>📊 {selected_symbol}</h2>
                <p style='color: rgba(255,255,255,0.8); margin: 0.25rem 0 0 0; font-size: 0.9rem;'>
                    Período: {selected_days} dias
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Obter dados com loading bonito
                with st.spinner(f"🔄 Carregando dados de {selected_symbol}..."):
                    stock_data = get_stock_data(selected_symbol, selected_days)
                
                if stock_data:
                    # Gráfico principal
                    st.markdown("### 📈 Histórico de Preços")
                    df = pd.DataFrame(stock_data['data'])
                    fig = create_candlestick_chart(df, selected_symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Indicadores técnicos em cards bonitos
                    st.markdown("### 📊 Indicadores Técnicos")
                    indicators = stock_data.get('indicators', {})
                    ind_cols = st.columns(4)
                    
                    with ind_cols[0]:
                        st.markdown("""
                        <div class='metric-card'>
                            <p style='color: #888; margin: 0; font-size: 0.85rem;'>MA 7 Dias</p>
                            <h3 style='margin: 0.25rem 0; color: #ffa502;'>$""" + f"{indicators.get('ma_7', 0):.2f}" + """</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ind_cols[1]:
                        st.markdown("""
                        <div class='metric-card'>
                            <p style='color: #888; margin: 0; font-size: 0.85rem;'>MA 30 Dias</p>
                            <h3 style='margin: 0.25rem 0; color: #3742fa;'>$""" + f"{indicators.get('ma_30', 0):.2f}" + """</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ind_cols[2]:
                        st.markdown("""
                        <div class='metric-card'>
                            <p style='color: #888; margin: 0; font-size: 0.85rem;'>Volatilidade</p>
                            <h3 style='margin: 0.25rem 0; color: #ff6348;'>$""" + f"{indicators.get('volatility', 0):.2f}" + """</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ind_cols[3]:
                        trend = indicators.get('trend', 'up')
                        trend_color = "#00ff88" if trend == 'up' else "#ff4757"
                        trend_icon = "📈" if trend == 'up' else "📉"
                        trend_text = "Alta" if trend == 'up' else "Baixa"
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p style='color: #888; margin: 0; font-size: 0.85rem;'>Tendência</p>
                            <h3 style='margin: 0.25rem 0; color: {trend_color};'>{trend_icon} {trend_text}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Não foi possível obter dados para {selected_symbol}")
                    st.info("💡 Tente novamente ou selecione outra ação")
            
            with col2:
                # Card de previsão estilizado
                st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                            padding: 1.5rem; border-radius: 1rem; border: 1px solid rgba(102, 126, 234, 0.3);'>
                    <h3 style='margin: 0 0 1rem 0;'>🔮 Previsão LSTM</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🚀 Fazer Previsão", use_container_width=True, type="primary"):
                    with st.spinner("🧠 Calculando previsão com modelo LSTM..."):
                        prediction = get_prediction(selected_symbol)
                    
                    if prediction:
                        st.success("✅ Previsão concluída!")
                        render_prediction_card(prediction)
                    else:
                        st.error("❌ Erro ao obter previsão")
                        st.info("💡 Tente novamente em alguns instantes")
                
                # Info do modelo em card
                st.markdown("---")
                st.markdown("""
                <div style='background: rgba(26, 26, 46, 0.5); padding: 1rem; border-radius: 0.75rem; border: 1px solid #333;'>
                    <h4 style='margin: 0 0 0.75rem 0; color: #667eea;'>🧠 Sobre o Modelo</h4>
                    <ul style='margin: 0; padding-left: 1.5rem; color: #aaa; font-size: 0.9rem;'>
                        <li><strong>Arquitetura:</strong> LSTM 2 camadas</li>
                        <li><strong>Features:</strong> 16 indicadores técnicos</li>
                        <li><strong>Período:</strong> 60 dias de histórico</li>
                        <li><strong>Hub:</strong> <a href='https://huggingface.co/henriquebap/stock-predictor-lstm' 
                            style='color: #667eea;' target='_blank'>HuggingFace 🤗</a></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Página inicial melhorada
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                        padding: 2rem; border-radius: 1rem; text-align: center; margin: 2rem 0;'>
                <h2 style='color: #667eea; margin: 0 0 1rem 0;'>👋 Bem-vindo ao Stock Predictor!</h2>
                <p style='color: #aaa; font-size: 1.1rem; margin: 0;'>
                    Selecione uma ação na barra lateral para começar<br>
                    ou clique em uma das ações populares abaixo
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar ações populares em grid bonito
            popular = get_popular_stocks()
            
            st.markdown("### 📋 Ações Mais Negociadas")
            st.markdown("<br>", unsafe_allow_html=True)
            
            for category, symbols in popular.get('categories', {}).items():
                st.markdown(f"""
                <div style='background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                            padding: 0.75rem 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                    <h4 style='margin: 0; color: #888;'>{category}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(min(len(symbols), 4))
                for i, sym in enumerate(symbols):
                    with cols[i % len(cols)]:
                        if st.button(f"📊 {sym}", key=f"pop_{sym}", use_container_width=True):
                            st.session_state['selected_symbol'] = sym
                            st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
    
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

