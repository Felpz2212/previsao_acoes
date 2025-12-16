"""
ML Health API Routes
Endpoints para monitoramento de saúde dos modelos ML
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
from datetime import datetime

from services.ml_health import ml_health_monitor

router = APIRouter(prefix="/api/ml-health", tags=["ML Health"])


@router.get("/health/{symbol}")
async def get_model_health(symbol: str):
    """
    Retorna métricas de saúde do modelo para um símbolo específico
    
    Métricas instantâneas - não requer ground truth futuro:
    - Health score (0-100)
    - Estatísticas de previsões
    - Alertas e recomendações
    
    Args:
        symbol: Símbolo da ação (ex: AAPL, GOOGL)
    
    Returns:
        Resumo de saúde do modelo
    """
    try:
        health = ml_health_monitor.get_model_health_summary(symbol.upper())
        
        if health.get('status') == 'no_data':
            return {
                "symbol": symbol.upper(),
                "status": "no_data",
                "message": f"Nenhuma previsão registrada para {symbol}",
                "timestamp": datetime.now().isoformat()
            }
        
        # Adicionar distribuição de previsões
        pred_dist = ml_health_monitor.check_prediction_distribution(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat(),
            "health_score": health['health_indicators']['score'],
            "status": health['health_indicators']['status'],
            "recommendation": health['health_indicators']['recommendation'],
            "metrics": {
                "prediction_stats": health['prediction_stats'],
                "prediction_distribution": pred_dist,
                "predictions_analyzed": health['predictions_count'],
                "time_range": health['time_range']
            },
            "alerts": health['health_indicators']['warnings']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular health: {str(e)}")


@router.get("/drift-report")
async def get_drift_report():
    """
    Relatório de data drift para todos os símbolos ativos
    
    Detecta mudanças na distribuição das features de entrada:
    - Z-score de cada feature vs baseline do treino
    - Severidade do drift (low/medium/high/critical)
    - Features outliers
    
    Returns:
        Relatório completo de drift
    """
    try:
        report = ml_health_monitor.get_drift_report()
        
        return {
            "timestamp": report['timestamp'],
            "summary": {
                "symbols_analyzed": report['symbols_analyzed'],
                "symbols_with_drift": report['symbols_with_drift'],
                "drift_ratio": report['drift_ratio']
            },
            "details": report['details'],
            "interpretation": {
                "drift_ratio < 0.2": "Normal - poucas features com drift",
                "drift_ratio 0.2-0.5": "Atenção - monitorar de perto",
                "drift_ratio > 0.5": "Crítico - possível mudança de regime"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar drift report: {str(e)}")


@router.get("/prediction-distribution/{symbol}")
async def get_prediction_distribution(symbol: str):
    """
    Análise da distribuição das previsões recentes
    
    Detecta mudanças no padrão de previsões:
    - % previsões positivas vs negativas
    - Magnitude média das previsões
    - Volatilidade das previsões
    - Previsões extremas
    
    Args:
        symbol: Símbolo da ação
    
    Returns:
        Métricas de distribuição
    """
    try:
        distribution = ml_health_monitor.check_prediction_distribution(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat(),
            "distribution": distribution
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar distribuição: {str(e)}")


@router.get("/data-quality")
async def get_data_quality_summary():
    """
    Resumo de qualidade de dados das últimas requisições
    
    Verifica:
    - Missing values
    - Valores inválidos
    - Outliers extremos
    
    Returns:
        Métricas de qualidade
    """
    try:
        # Analisar últimas N features registradas
        recent = list(ml_health_monitor.recent_features)[-100:]
        
        if not recent:
            return {
                "status": "no_data",
                "message": "Nenhuma feature registrada ainda"
            }
        
        # Calcular qualidade para cada
        quality_scores = []
        issues_summary = []
        
        for entry in recent:
            quality = ml_health_monitor.check_data_quality(entry['features'])
            quality_scores.append(quality['quality_score'])
            
            if quality['has_issues']:
                issues_summary.extend(quality['issues'])
        
        # Agregar
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Contar issues mais comuns
        from collections import Counter
        issue_counts = Counter(issues_summary)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "samples_analyzed": len(recent),
            "average_quality_score": round(avg_quality, 1),
            "quality_status": "good" if avg_quality >= 80 else "warning" if avg_quality >= 60 else "poor",
            "common_issues": [
                {"issue": issue, "count": count} 
                for issue, count in issue_counts.most_common(5)
            ] if issue_counts else []
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar qualidade: {str(e)}")


@router.get("/overview")
async def get_ml_health_overview():
    """
    Visão geral de saúde de todos os modelos
    
    Agrega métricas de:
    - Todos os símbolos ativos
    - Health scores
    - Drift status
    - Qualidade de dados
    
    Returns:
        Dashboard completo de saúde ML
    """
    try:
        # Símbolos ativos
        active_symbols = set(p['symbol'] for p in ml_health_monitor.recent_predictions)
        
        if not active_symbols:
            return {
                "status": "no_data",
                "message": "Nenhum modelo com previsões recentes",
                "timestamp": datetime.now().isoformat()
            }
        
        # Coletar health de cada símbolo
        models_health = {}
        total_score = 0
        status_counts = {"healthy": 0, "warning": 0, "poor": 0, "critical": 0}
        
        for symbol in active_symbols:
            health = ml_health_monitor.get_model_health_summary(symbol)
            
            if health.get('status') != 'no_data':
                score = health['health_indicators']['score']
                status = health['health_indicators']['status']
                
                models_health[symbol] = {
                    "score": score,
                    "status": status,
                    "predictions_count": health['predictions_count']
                }
                
                total_score += score
                status_counts[status] = status_counts.get(status, 0) + 1
        
        # Drift report
        drift_report = ml_health_monitor.get_drift_report()
        
        # Calcular médias
        avg_score = total_score / len(models_health) if models_health else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_models": len(active_symbols),
                "average_health_score": round(avg_score, 1),
                "status_distribution": status_counts,
                "models_with_drift": drift_report['symbols_with_drift']
            },
            "models": models_health,
            "drift_summary": {
                "symbols_analyzed": drift_report['symbols_analyzed'],
                "symbols_with_drift": drift_report['symbols_with_drift'],
                "drift_ratio": drift_report['drift_ratio']
            },
            "overall_status": (
                "healthy" if avg_score >= 80 else 
                "warning" if avg_score >= 60 else 
                "poor"
            )
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar overview: {str(e)}")
