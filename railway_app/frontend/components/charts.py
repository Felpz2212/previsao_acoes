"""
Charts Component - Graficos interativos com Plotly
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict


def create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Cria grafico de candlestick com volume."""
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['ma_7'] = df['close'].rolling(7).mean()
    df['ma_30'] = df['close'].rolling(30).mean()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='OHLC',
            increasing_line_color='#00ff88', decreasing_line_color='#ff4757'
        ), row=1, col=1
    )
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma_7'], name='MA 7', line=dict(color='#ffa502', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma_30'], name='MA 30', line=dict(color='#3742fa', width=1.5)), row=1, col=1)
    
    colors = ['#00ff88' if c >= o else '#ff4757' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors, opacity=0.7), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark', height=600, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_rangeslider_visible=False, margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.8)'
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(tickprefix='$', row=1, col=1)
    
    return fig


def create_comparison_chart(data: Dict[str, dict]) -> go.Figure:
    """Cria grafico de comparacao normalizado."""
    fig = go.Figure()
    colors = ['#667eea', '#00ff88', '#ff4757', '#ffa502', '#3742fa']
    
    for i, (symbol, info) in enumerate(data.items()):
        df = pd.DataFrame(info['data'])
        base_price = df['close'].iloc[0]
        normalized = (df['close'] / base_price) * 100
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=normalized, name=f"{symbol}",
            line=dict(color=colors[i % len(colors)], width=2), mode='lines'
        ))
    
    fig.add_hline(y=100, line_dash='dash', line_color='white', opacity=0.5)
    fig.update_layout(
        template='plotly_dark', height=500, title='Comparacao (Base 100)',
        showlegend=True, xaxis_title='Data', yaxis_title='Performance (%)',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,26,46,0.8)', hovermode='x unified'
    )
    
    return fig

