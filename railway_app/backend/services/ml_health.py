"""
ML Health Monitoring - Métricas instantâneas de saúde do modelo
Sem necessidade de ground truth futuro
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
from collections import deque
import json
import os
from pathlib import Path
from loguru import logger


@dataclass
class FeatureStats:
    """Estatísticas de uma feature do treino"""
    mean: float
    std: float
    min: float
    max: float


@dataclass
class DriftMetrics:
    """Métricas de drift calculadas"""
    feature: str
    z_score: float
    is_outlier: bool
    severity: str  # 'low', 'medium', 'high', 'critical'


class MLHealthMonitoring:
    """
    Monitoramento de saúde do modelo ML
    Métricas instantâneas sem necessidade de valores futuros
    """
    
    def __init__(self, models_path: str = "models"):
        self.models_path = Path(models_path)
        
        # Carregar estatísticas do treino
        self.baseline_stats = self._load_training_stats()
        
        # Histórico recente de previsões (últimas 1000)
        self.recent_predictions = deque(maxlen=1000)
        
        # Histórico de features (últimas 1000)
        self.recent_features = deque(maxlen=1000)
        
        logger.info("🧠 MLHealthMonitoring inicializado")
    
    def _load_training_stats(self) -> Dict[str, Dict[str, FeatureStats]]:
        """Carrega estatísticas do treino dos arquivos metadata"""
        stats = {}
        
        # Procurar por arquivos metadata_*.json
        if not self.models_path.exists():
            logger.warning(f"Models path não existe: {self.models_path}")
            return stats
        
        for metadata_file in self.models_path.glob("metadata_*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Extrair símbolo do nome do arquivo
                symbol = metadata_file.stem.replace("metadata_", "")
                
                # Extrair estatísticas das features se existirem
                if 'training_data' in metadata:
                    training_data = metadata['training_data']
                    feature_stats = {}
                    
                    # Criar stats básicas (você pode expandir isso)
                    # Por enquanto, criar valores padrão baseados no tipo de ação
                    feature_stats['volatility'] = FeatureStats(
                        mean=2.5, std=0.8, min=0.1, max=10.0
                    )
                    feature_stats['volume'] = FeatureStats(
                        mean=1e6, std=5e5, min=1e4, max=1e8
                    )
                    feature_stats['ma_7'] = FeatureStats(
                        mean=150.0, std=50.0, min=50.0, max=500.0
                    )
                    feature_stats['ma_30'] = FeatureStats(
                        mean=150.0, std=50.0, min=50.0, max=500.0
                    )
                    
                    stats[symbol] = feature_stats
                    logger.info(f"📊 Estatísticas carregadas para {symbol}")
            
            except Exception as e:
                logger.warning(f"Erro ao carregar {metadata_file}: {e}")
        
        return stats
    
    def log_prediction(self, symbol: str, prediction: float, features: Optional[Dict] = None):
        """
        Registra uma previsão feita pelo modelo
        
        Args:
            symbol: Símbolo da ação
            prediction: Valor previsto (% de mudança)
            features: Features usadas (opcional)
        """
        self.recent_predictions.append({
            'symbol': symbol,
            'prediction': prediction,
            'timestamp': datetime.now(),
            'abs_prediction': abs(prediction)
        })
        
        if features:
            self.recent_features.append({
                'symbol': symbol,
                'features': features,
                'timestamp': datetime.now()
            })
    
    def check_feature_drift(self, symbol: str, features: Dict[str, float]) -> List[DriftMetrics]:
        """
        Verifica drift nas features de entrada
        Compara com estatísticas do treino
        
        Args:
            symbol: Símbolo da ação
            features: Features atuais
            
        Returns:
            Lista de features com drift detectado
        """
        drifts = []
        
        baseline = self.baseline_stats.get(symbol, {})
        if not baseline:
            logger.warning(f"Sem baseline stats para {symbol}")
            return drifts
        
        for feature_name, current_value in features.items():
            if feature_name not in baseline:
                continue
            
            if current_value is None or np.isnan(current_value):
                continue
            
            stats = baseline[feature_name]
            
            # Calcular z-score
            z_score = abs(current_value - stats.mean) / stats.std if stats.std > 0 else 0
            
            # Verificar se é outlier
            is_outlier = z_score > 3
            
            # Determinar severidade
            if z_score < 2:
                severity = 'low'
            elif z_score < 3:
                severity = 'medium'
            elif z_score < 4:
                severity = 'high'
            else:
                severity = 'critical'
            
            # Reportar se significativo (> 2σ)
            if z_score > 2:
                drifts.append(DriftMetrics(
                    feature=feature_name,
                    z_score=round(z_score, 2),
                    is_outlier=is_outlier,
                    severity=severity
                ))
        
        return drifts
    
    def check_prediction_distribution(self, symbol: str) -> Dict:
        """
        Analisa distribuição das previsões recentes
        Detecta mudanças no padrão de previsões
        
        Args:
            symbol: Símbolo da ação
            
        Returns:
            Métricas de distribuição
        """
        # Filtrar previsões do símbolo
        symbol_preds = [
            p['prediction'] for p in self.recent_predictions 
            if p['symbol'] == symbol
        ]
        
        if len(symbol_preds) < 30:
            return {
                'status': 'insufficient_data',
                'sample_size': len(symbol_preds),
                'message': 'Mínimo 30 previsões necessárias'
            }
        
        predictions_array = np.array(symbol_preds)
        
        # Calcular métricas
        positive_count = np.sum(predictions_array > 0)
        positive_ratio = positive_count / len(predictions_array)
        
        mean_magnitude = np.mean(np.abs(predictions_array))
        prediction_volatility = np.std(predictions_array)
        
        # Contar previsões extremas (> 2σ da média)
        extreme_threshold = mean_magnitude + 2 * prediction_volatility
        extreme_count = np.sum(np.abs(predictions_array) > extreme_threshold)
        
        # Gerar alertas
        alerts = []
        
        # Muito enviesado para um lado
        if positive_ratio > 0.85:
            alerts.append('⚠️ 85%+ previsões positivas - possível viés bullish')
        elif positive_ratio < 0.15:
            alerts.append('⚠️ 85%+ previsões negativas - possível viés bearish')
        
        # Muitas previsões extremas
        extreme_ratio = extreme_count / len(predictions_array)
        if extreme_ratio > 0.1:
            alerts.append(f'⚠️ {extreme_ratio*100:.1f}% previsões extremas - alta incerteza')
        
        # Volatilidade muito alta
        if prediction_volatility > mean_magnitude * 1.5:
            alerts.append('⚠️ Volatilidade de previsões muito alta')
        
        return {
            'status': 'ok',
            'sample_size': len(symbol_preds),
            'positive_ratio': round(positive_ratio, 3),
            'positive_count': int(positive_count),
            'negative_count': int(len(symbol_preds) - positive_count),
            'mean_magnitude': round(mean_magnitude, 3),
            'volatility': round(prediction_volatility, 3),
            'extreme_predictions': int(extreme_count),
            'extreme_ratio': round(extreme_ratio, 3),
            'alerts': alerts
        }
    
    def check_data_quality(self, features: Dict[str, float]) -> Dict:
        """
        Verifica qualidade dos dados de entrada
        
        Args:
            features: Features para verificar
            
        Returns:
            Relatório de qualidade
        """
        issues = []
        warnings = []
        
        # Missing values
        missing = [k for k, v in features.items() 
                  if v is None or (isinstance(v, float) and np.isnan(v))]
        if missing:
            issues.append(f'Missing values: {", ".join(missing)}')
        
        # Valores negativos onde não deveria
        if features.get('volume', 0) < 0:
            issues.append('Volume negativo detectado')
        
        if features.get('close', 0) <= 0:
            issues.append('Preço de fechamento inválido')
        
        # Volatilidade extrema
        volatility = features.get('volatility', 0)
        if volatility > 50:
            warnings.append(f'Volatilidade muito alta: {volatility:.2f}%')
        elif volatility < 0:
            issues.append('Volatilidade negativa')
        
        # MAs invertidas (geralmente MA7 próxima de MA30)
        ma7 = features.get('ma_7', 0)
        ma30 = features.get('ma_30', 0)
        if ma7 > 0 and ma30 > 0:
            diff_pct = abs(ma7 - ma30) / ma30 * 100
            if diff_pct > 50:
                warnings.append(f'MAs muito distantes: {diff_pct:.1f}% diferença')
        
        # Calcular score de qualidade
        quality_score = 100
        quality_score -= len(issues) * 25  # Cada issue -25%
        quality_score -= len(warnings) * 10  # Cada warning -10%
        quality_score = max(0, min(100, quality_score))
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'warnings': warnings,
            'quality_score': quality_score,
            'status': 'good' if quality_score >= 80 else 'warning' if quality_score >= 60 else 'poor'
        }
    
    def get_model_health_summary(self, symbol: str) -> Dict:
        """
        Resumo geral da saúde do modelo para um símbolo
        
        Args:
            symbol: Símbolo da ação
            
        Returns:
            Resumo de saúde
        """
        # Filtrar previsões recentes deste símbolo
        recent_symbol = [
            p for p in self.recent_predictions 
            if p['symbol'] == symbol
        ][-100:]  # Últimas 100
        
        if not recent_symbol:
            return {
                'status': 'no_data',
                'message': f'Nenhuma previsão registrada para {symbol}'
            }
        
        # Extrair valores
        predictions = [p['prediction'] for p in recent_symbol]
        predictions_array = np.array(predictions)
        
        # Calcular health score
        health_indicators = self._calculate_health_score(predictions)
        
        return {
            'symbol': symbol,
            'predictions_count': len(predictions),
            'time_range': {
                'start': recent_symbol[0]['timestamp'].isoformat(),
                'end': recent_symbol[-1]['timestamp'].isoformat()
            },
            'prediction_stats': {
                'mean': round(float(np.mean(predictions)), 3),
                'std': round(float(np.std(predictions)), 3),
                'min': round(float(np.min(predictions)), 3),
                'max': round(float(np.max(predictions)), 3),
                'positive_ratio': round(float(np.sum(predictions_array > 0) / len(predictions)), 3)
            },
            'health_indicators': health_indicators
        }
    
    def _calculate_health_score(self, predictions: List[float]) -> Dict:
        """
        Calcula score de saúde 0-100 baseado nas previsões
        
        Args:
            predictions: Lista de previsões
            
        Returns:
            Score e status
        """
        score = 100
        warnings = []
        
        predictions_array = np.array(predictions)
        
        # 1. Verificar viés (muito enviesado para um lado)
        positive_ratio = np.sum(predictions_array > 0) / len(predictions)
        if positive_ratio > 0.9:
            score -= 30
            warnings.append(f'Muito enviesado positivo: {positive_ratio*100:.1f}%')
        elif positive_ratio < 0.1:
            score -= 30
            warnings.append(f'Muito enviesado negativo: {positive_ratio*100:.1f}%')
        elif positive_ratio > 0.8 or positive_ratio < 0.2:
            score -= 15
            warnings.append(f'Enviesado: {positive_ratio*100:.1f}% positivas')
        
        # 2. Verificar volatilidade das previsões
        volatility = np.std(predictions)
        mean_abs = np.mean(np.abs(predictions))
        
        if mean_abs > 0:
            vol_ratio = volatility / mean_abs
            if vol_ratio > 2:
                score -= 20
                warnings.append('Previsões muito voláteis')
            elif vol_ratio > 1.5:
                score -= 10
                warnings.append('Previsões com alta volatilidade')
        
        # 3. Verificar valores extremos
        if mean_abs > 0:
            extreme_threshold = mean_abs + 3 * volatility
            extreme_ratio = np.sum(np.abs(predictions) > extreme_threshold) / len(predictions)
            
            if extreme_ratio > 0.1:
                score -= 15
                warnings.append(f'{extreme_ratio*100:.1f}% previsões extremas')
            elif extreme_ratio > 0.05:
                score -= 5
        
        # 4. Verificar se todas previsões são muito pequenas ou grandes
        if mean_abs < 0.1:
            warnings.append('Previsões muito conservadoras (< 0.1%)')
        elif mean_abs > 10:
            score -= 10
            warnings.append('Previsões muito agressivas (> 10%)')
        
        # Garantir score entre 0-100
        score = max(0, min(100, score))
        
        # Determinar status
        if score >= 80:
            status = 'healthy'
        elif score >= 60:
            status = 'warning'
        elif score >= 40:
            status = 'poor'
        else:
            status = 'critical'
        
        return {
            'score': int(score),
            'status': status,
            'warnings': warnings,
            'recommendation': self._get_recommendation(score, warnings)
        }
    
    def _get_recommendation(self, score: int, warnings: List[str]) -> str:
        """Gera recomendação baseada no score e warnings"""
        if score >= 80:
            return 'Modelo operando normalmente'
        elif score >= 60:
            return 'Monitorar de perto - possíveis problemas detectados'
        elif score >= 40:
            return 'Investigar warnings - pode precisar re-treino'
        else:
            return 'Ação necessária - modelo pode estar comprometido'
    
    def get_drift_report(self) -> Dict:
        """
        Relatório de drift para todos os símbolos ativos
        
        Returns:
            Relatório completo de drift
        """
        # Símbolos com previsões recentes
        active_symbols = set(p['symbol'] for p in self.recent_predictions)
        
        report = {}
        symbols_with_drift = 0
        
        for symbol in active_symbols:
            # Pegar features mais recentes deste símbolo
            recent_features = [
                f for f in self.recent_features 
                if f['symbol'] == symbol
            ]
            
            if not recent_features:
                continue
            
            latest = recent_features[-1]['features']
            drifts = self.check_feature_drift(symbol, latest)
            
            if drifts:
                symbols_with_drift += 1
                report[symbol] = {
                    'has_drift': True,
                    'drift_count': len(drifts),
                    'drifts': [
                        {
                            'feature': d.feature,
                            'z_score': d.z_score,
                            'severity': d.severity,
                            'is_outlier': d.is_outlier
                        } for d in drifts
                    ],
                    'timestamp': recent_features[-1]['timestamp'].isoformat()
                }
            else:
                report[symbol] = {
                    'has_drift': False,
                    'message': 'Todas features dentro do esperado'
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(active_symbols),
            'symbols_with_drift': symbols_with_drift,
            'drift_ratio': round(symbols_with_drift / len(active_symbols), 3) if active_symbols else 0,
            'details': report
        }


# Instância global
ml_health_monitor = MLHealthMonitoring()
