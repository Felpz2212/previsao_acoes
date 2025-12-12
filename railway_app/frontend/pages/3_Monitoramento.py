"""
Página de Monitoramento - Métricas da API em tempo real.
"""
import streamlit as st
import requests
import os
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Configuração
API_URL = os.getenv("API_URL", "http://localhost:8000")
if not API_URL.startswith("http"):
    API_URL = f"https://{API_URL}"

# Grafana Cloud embed URL (configurável)
GRAFANA_EMBED_URL = os.getenv(
    "GRAFANA_EMBED_URL", 
    "https://henriquebap.grafana.net/public-dashboards/"  # Será preenchido depois
)

st.set_page_config(
    page_title="Monitoramento | Stock Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Monitoramento em Tempo Real")
st.markdown("Métricas de performance da API e dos modelos LSTM")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Métricas da API", "🧠 Performance dos Modelos", "🌐 Grafana Dashboard"])


def fetch_monitoring_data():
    """Busca dados de monitoramento da API."""
    try:
        response = requests.get(f"{API_URL}/api/monitoring", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Erro ao buscar métricas: {e}")
    return None


def fetch_recent_requests():
    """Busca requisições recentes."""
    try:
        response = requests.get(f"{API_URL}/api/monitoring/requests?limit=50", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


with tab1:
    st.subheader("📈 Métricas da API")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Atualizar", key="refresh_api"):
            st.rerun()
    
    data = fetch_monitoring_data()
    
    if data:
        # KPIs principais
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "⏱️ Uptime",
                data.get('uptime_human', 'N/A'),
                delta=None
            )
        
        with col2:
            st.metric(
                "📨 Total Requests",
                f"{data.get('total_requests', 0):,}",
                delta=f"{data.get('requests_per_minute', 0):.1f}/min"
            )
        
        with col3:
            error_rate = data.get('error_rate_percent', 0)
            st.metric(
                "❌ Taxa de Erros",
                f"{error_rate:.1f}%",
                delta=f"-{error_rate:.1f}%" if error_rate < 5 else f"+{error_rate:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "🔮 Total Previsões",
                f"{data.get('total_predictions', 0):,}"
            )
        
        with col5:
            system = data.get('system', {})
            cpu = system.get('cpu_percent', 0) if system else 0
            st.metric(
                "💻 CPU",
                f"{cpu:.1f}%",
                delta="OK" if cpu < 80 else "Alto"
            )
        
        st.divider()
        
        # Métricas por endpoint
        st.subheader("📊 Performance por Endpoint")
        
        endpoints = data.get('endpoints', {})
        if endpoints:
            endpoint_data = []
            for endpoint, stats in endpoints.items():
                endpoint_data.append({
                    'Endpoint': endpoint,
                    'Requests': stats.get('count', 0),
                    'Avg (ms)': stats.get('avg_time_ms', 0),
                    'Min (ms)': stats.get('min_time_ms', 0),
                    'Max (ms)': stats.get('max_time_ms', 0),
                    'Error %': stats.get('error_rate', 0)
                })
            
            df = pd.DataFrame(endpoint_data)
            df = df.sort_values('Requests', ascending=False)
            
            # Gráfico de barras
            fig = px.bar(
                df.head(10),
                x='Endpoint',
                y='Avg (ms)',
                color='Error %',
                color_continuous_scale='RdYlGn_r',
                title='Latência Média por Endpoint (Top 10)'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum endpoint registrado ainda. Faça algumas requisições!")
        
        # Sistema
        st.subheader("💻 Recursos do Sistema")
        
        system = data.get('system')
        if system:
            col1, col2 = st.columns(2)
            
            with col1:
                # CPU gauge
                fig_cpu = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=system.get('cpu_percent', 0),
                    title={'text': "CPU (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ]
                    }
                ))
                fig_cpu.update_layout(height=250)
                st.plotly_chart(fig_cpu, use_container_width=True)
            
            with col2:
                # Memory gauge
                fig_mem = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=system.get('memory_percent', 0),
                    title={'text': "Memória (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgreen"},
                            {'range': [60, 85], 'color': "yellow"},
                            {'range': [85, 100], 'color': "red"}
                        ]
                    }
                ))
                fig_mem.update_layout(height=250)
                st.plotly_chart(fig_mem, use_container_width=True)
            
            st.info(f"💾 Memória: {system.get('memory_used_mb', 0):.0f} MB usados / {system.get('memory_available_mb', 0):.0f} MB disponíveis")
    else:
        st.warning("⚠️ Não foi possível conectar ao serviço de monitoramento")
        st.code(f"API URL: {API_URL}/api/monitoring")


with tab2:
    st.subheader("🧠 Performance dos Modelos")
    
    data = fetch_monitoring_data()
    
    if data:
        models = data.get('models', {})
        
        if models:
            st.markdown("### Tempo de Inferência por Modelo")
            
            model_data = []
            for symbol, stats in models.items():
                model_data.append({
                    'Símbolo': symbol,
                    'Previsões': stats.get('predictions', 0),
                    'Avg (ms)': stats.get('avg_inference_ms', 0),
                    'Min (ms)': stats.get('min_inference_ms', 0),
                    'Max (ms)': stats.get('max_inference_ms', 0)
                })
            
            df = pd.DataFrame(model_data)
            df = df.sort_values('Previsões', ascending=False)
            
            # Gráfico
            fig = px.bar(
                df,
                x='Símbolo',
                y='Avg (ms)',
                color='Previsões',
                color_continuous_scale='Blues',
                title='Tempo Médio de Inferência por Modelo'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("🔮 Faça algumas previsões para ver as métricas dos modelos!")
        
        # Métricas de qualidade (se disponíveis)
        st.markdown("### 📊 Métricas de Qualidade dos Modelos")
        
        quality_data = [
            {"Símbolo": "MSFT", "MAPE": 3.47, "R²": 0.83, "Dir. Acc": 54.0},
            {"Símbolo": "V", "MAPE": 3.72, "R²": -0.77, "Dir. Acc": 50.0},
            {"Símbolo": "TSLA", "MAPE": 5.61, "R²": 0.88, "Dir. Acc": 46.6},
            {"Símbolo": "GOOGL", "MAPE": 7.36, "R²": 0.85, "Dir. Acc": 55.7},
            {"Símbolo": "NVDA", "MAPE": 7.50, "R²": 0.81, "Dir. Acc": 46.0},
            {"Símbolo": "META", "MAPE": 7.60, "R²": 0.42, "Dir. Acc": 55.7},
            {"Símbolo": "AAPL", "MAPE": 8.28, "R²": 0.04, "Dir. Acc": 52.3},
            {"Símbolo": "JPM", "MAPE": 10.42, "R²": -0.28, "Dir. Acc": 49.4},
            {"Símbolo": "AMZN", "MAPE": 11.61, "R²": -1.32, "Dir. Acc": 51.7},
        ]
        
        df_quality = pd.DataFrame(quality_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_quality,
                x='Símbolo',
                y='MAPE',
                color='MAPE',
                color_continuous_scale='RdYlGn_r',
                title='MAPE por Modelo (%)'
            )
            fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Limite aceitável")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df_quality,
                x='Símbolo',
                y='Dir. Acc',
                color='Dir. Acc',
                color_continuous_scale='RdYlGn',
                title='Acurácia Direcional (%)'
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Random")
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_quality, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Não foi possível conectar ao serviço de monitoramento")


with tab3:
    st.subheader("🌐 Grafana Dashboard")
    
    st.markdown("""
    ### Opções de Visualização
    
    O monitoramento completo está disponível no **Grafana Cloud**.
    """)
    
    # Link direto para o Grafana
    grafana_url = "https://henriquebap.grafana.net"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        #### 🔗 Acesso Direto
        
        - **Grafana Cloud**: [{grafana_url}]({grafana_url})
        - **Usuário**: henriquebap
        """)
        
        if st.button("🌐 Abrir Grafana Cloud", type="primary"):
            st.markdown(f'<meta http-equiv="refresh" content="0;url={grafana_url}">', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        #### 📊 Métricas Disponíveis
        
        - `stock_predictor_requests_total`
        - `stock_predictor_request_duration_seconds`
        - `stock_predictor_predictions_total`
        - `stock_predictor_model_inference_seconds`
        - `stock_predictor_system_cpu_percent`
        - `stock_predictor_system_memory_percent`
        """)
    
    st.divider()
    
    # Endpoint de métricas raw
    st.markdown("### 📝 Métricas Raw (Prometheus Format)")
    
    if st.button("🔄 Carregar Métricas"):
        try:
            response = requests.get(f"{API_URL}/metrics", timeout=10)
            if response.status_code == 200:
                st.code(response.text[:3000] + "\n...", language="text")
                st.success(f"✅ Métricas carregadas de {API_URL}/metrics")
            else:
                st.error(f"Erro: {response.status_code}")
        except Exception as e:
            st.error(f"Erro ao carregar métricas: {e}")
    
    st.divider()
    
    # Instruções para configurar Grafana Cloud
    with st.expander("📖 Como configurar Grafana Cloud"):
        st.markdown("""
        ### Passo 1: Criar um Dashboard
        
        1. Acesse https://henriquebap.grafana.net
        2. Vá em **Dashboards** → **New** → **Import**
        3. Cole o JSON do dashboard (disponível em `monitoring/grafana/provisioning/dashboards/stock-predictor.json`)
        
        ### Passo 2: Configurar Data Source
        
        Para coletar métricas da API em produção:
        
        1. Vá em **Connections** → **Data Sources** → **Add**
        2. Selecione **Prometheus**
        3. URL: `https://previsaoacoes-back-production.up.railway.app`
        4. **Ou** use Grafana Agent para scraping
        
        ### Passo 3: Tornar Dashboard Público
        
        1. Abra o dashboard
        2. Clique em **Share** → **Public Dashboard**
        3. Ative **Public Dashboard**
        4. Copie o link gerado
        """)


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    📊 Stock Predictor - Monitoramento em Tempo Real<br>
    Tech Challenge Fase 4 - FIAP Pós-Tech ML Engineering
</div>
""", unsafe_allow_html=True)

