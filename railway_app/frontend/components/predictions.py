"""
Predictions Component - Exibicao de previsoes LSTM
Layout responsivo para telas menores
"""
import streamlit as st


def render_prediction_card(prediction: dict):
    """
    Renderiza card de previsao com layout responsivo.
    """
    symbol = prediction.get('symbol', '')
    current = prediction.get('current_price', 0)
    predicted = prediction.get('predicted_price', 0)
    change = prediction.get('change_percent', 0)
    direction = prediction.get('direction', '')
    confidence = prediction.get('confidence', 'Moderada')
    model_type = prediction.get('model_type', 'LSTM')
    indicators = prediction.get('indicators', {})
    
    # Determinar cor
    if change > 0:
        delta_color = "normal"
        emoji = "📈"
    else:
        delta_color = "inverse"
        emoji = "📉"
    
    # Titulo
    st.markdown(f"### {emoji} {symbol}")
    
    # Layout em 2 linhas para caber melhor
    # Linha 1: Preço atual e previsão
    st.metric(
        label="💰 Atual → 🔮 Previsto",
        value=f"${predicted:.2f}",
        delta=f"{change:+.2f}% (de ${current:.2f})",
        delta_color=delta_color
    )
    
    # Linha 2: Direção
    dir_text = direction.replace("📈 ", "").replace("📉 ", "")
    st.markdown(f"**Direção:** {emoji} {dir_text}")
    
    st.divider()
    
    # Info do modelo (compacto)
    st.markdown(f"**Modelo:** {model_type}")
    st.markdown(f"**Confiança:** {confidence}")
    
    # Indicadores (compacto)
    if indicators:
        st.divider()
        ma7 = indicators.get('ma_7', 0)
        ma30 = indicators.get('ma_30', 0)
        trend = indicators.get('trend', 'bullish')
        trend_emoji = "📈" if trend == 'bullish' else "📉"
        
        st.markdown(f"""
        **Indicadores:**  
        MA7: ${ma7:.2f} | MA30: ${ma30:.2f} | {trend_emoji} {trend.title()}
        """)
    
    # Disclaimer
    st.caption("⚠️ Previsão educacional apenas!")


def render_prediction_card_expanded(prediction: dict):
    """
    Versao expandida do card para telas maiores.
    """
    symbol = prediction.get('symbol', '')
    current = prediction.get('current_price', 0)
    predicted = prediction.get('predicted_price', 0)
    change = prediction.get('change_percent', 0)
    direction = prediction.get('direction', '')
    confidence = prediction.get('confidence', 'Moderada')
    model_type = prediction.get('model_type', 'LSTM')
    indicators = prediction.get('indicators', {})
    
    if change > 0:
        delta_color = "normal"
        emoji = "📈"
    else:
        delta_color = "inverse"
        emoji = "📉"
    
    st.markdown(f"### {emoji} Previsão {symbol}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("💰 Preço Atual", f"${current:.2f}")
    
    with col2:
        st.metric(
            "🔮 Previsão",
            f"${predicted:.2f}",
            f"{change:+.2f}%",
            delta_color=delta_color
        )
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.info(f"🧠 **Modelo:** {model_type}")
    
    with col4:
        st.info(f"🎯 **Confiança:** {confidence}")
    
    if indicators:
        st.markdown("#### 📊 Indicadores")
        
        icol1, icol2, icol3 = st.columns(3)
        
        with icol1:
            st.metric("MA 7", f"${indicators.get('ma_7', 0):.2f}")
        with icol2:
            st.metric("MA 30", f"${indicators.get('ma_30', 0):.2f}")
        with icol3:
            trend = indicators.get('trend', 'bullish')
            st.metric("Tendência", "📈 Alta" if trend == 'bullish' else "📉 Baixa")
    
    st.caption("⚠️ Previsão educacional. NÃO use para investimentos!")
