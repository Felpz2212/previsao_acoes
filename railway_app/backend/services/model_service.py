"""
Model Service - Gerenciamento de modelos LSTM
Deploy auto-contido para Railway

Suporta:
- Modelos originais (LSTMPredictor)
- Modelos melhorados (ImprovedLSTMPredictor) com Attention
"""
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import torch
from loguru import logger

# Importar de core/ (copia local para deploy)
from core.lstm_model import LSTMPredictor
from core.improved_lstm import ImprovedLSTMPredictor, detect_model_type
from core.preprocessor import StockDataPreprocessor

# HuggingFace Hub para baixar modelos
from huggingface_hub import hf_hub_download


class ModelService:
    """Servico de gerenciamento de modelos LSTM."""
    
    HUB_REPO = "henriquebap/stock-predictor-lstm"
    LOCAL_CACHE = Path("models")
    
    # Modelos disponíveis no Hub (atualizados)
    AVAILABLE_MODELS = [
        "BASE", "AAPL", "GOOGL", "MSFT", "AMZN", 
        "META", "NVDA", "TSLA", "JPM", "V"
    ]
    
    def __init__(self):
        self.model_cache: Dict[str, dict] = {}
        self.LOCAL_CACHE.mkdir(exist_ok=True)
        logger.info(f"🧠 ModelService inicializado | Hub: {self.HUB_REPO}")
    
    def _download_from_hub(self, filename: str) -> Path:
        """Baixa arquivo do HuggingFace Hub."""
        logger.info(f"📥 Baixando do Hub: {filename}")
        path = Path(hf_hub_download(
            repo_id=self.HUB_REPO,
            filename=filename,
            cache_dir=str(self.LOCAL_CACHE / "hub_cache")
        ))
        logger.info(f"✅ Download concluido: {filename}")
        return path
    
    def _load_model(self, symbol: str) -> Optional[dict]:
        """Carrega modelo do cache local ou Hub."""
        model_file = f"lstm_model_{symbol}.pth"
        scaler_file = f"scaler_{symbol}.pkl"
        
        logger.info(f"🔍 Procurando modelo para {symbol}...")
        
        # Tentar cache local primeiro
        local_model = self.LOCAL_CACHE / model_file
        local_scaler = self.LOCAL_CACHE / scaler_file
        
        if local_model.exists() and local_scaler.exists():
            model_path = local_model
            scaler_path = local_scaler
            source = "local"
            logger.info(f"📁 Modelo LOCAL encontrado para {symbol}")
        else:
            # Tentar HuggingFace Hub - modelo especifico
            logger.info(f"🌐 Buscando modelo {symbol} no HuggingFace Hub...")
            try:
                model_path = self._download_from_hub(model_file)
                scaler_path = self._download_from_hub(scaler_file)
                source = "hub"
                logger.info(f"✅ Modelo para {symbol} encontrado no Hub!")
            except Exception as e:
                # Fallback para modelo BASE
                logger.warning(f"⚠️ Modelo específico para {symbol} não encontrado: {e}")
                logger.info(f"🔄 Usando modelo BASE genérico...")
                try:
                    model_path = self._download_from_hub("lstm_model_BASE.pth")
                    scaler_path = self._download_from_hub("scaler_BASE.pkl")
                    source = "base"
                    logger.info(f"✅ Modelo BASE carregado para {symbol}")
                except Exception as e2:
                    logger.error(f"❌ Falha ao carregar modelo BASE: {e2}")
                    return None
        
        # Detectar tipo do modelo e carregar
        try:
            model_type = detect_model_type(model_path)
            logger.info(f"🔎 Tipo detectado: {model_type}")
            
            if model_type == 'improved':
                model = ImprovedLSTMPredictor.load(model_path)
                logger.info(f"✅ Carregado como ImprovedLSTMPredictor")
            else:
                model = LSTMPredictor.load(model_path)
                logger.info(f"✅ Carregado como LSTMPredictor")
            
            preprocessor = StockDataPreprocessor.load(scaler_path)
            
            logger.info(f"🎯 Modelo carregado | Symbol: {symbol} | Source: {source} | Type: {model_type}")
            
            return {
                'model': model,
                'preprocessor': preprocessor,
                'source': source,
                'model_type': model_type,
                'symbol_requested': symbol
            }
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return None
    
    def get_model(self, symbol: str) -> Optional[dict]:
        """Obtem modelo do cache ou carrega."""
        symbol = symbol.upper()
        
        if symbol not in self.model_cache:
            logger.info(f"📦 Modelo {symbol} não está em cache, carregando...")
            model_data = self._load_model(symbol)
            if model_data:
                self.model_cache[symbol] = model_data
        else:
            logger.info(f"⚡ Modelo {symbol} encontrado em cache!")
        
        return self.model_cache.get(symbol)
    
    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        """Faz previsao para um simbolo."""
        logger.info(f"🔮 Iniciando previsão para {symbol}...")
        
        # Tentar modelo especifico, depois BASE
        model_data = self.get_model(symbol)
        
        if not model_data:
            logger.warning(f"⚠️ Nenhum modelo disponível para {symbol}, tentando BASE...")
            model_data = self.get_model("BASE")
        
        if not model_data:
            # Fallback: media movel simples
            logger.warning(f"⚠️ Usando FALLBACK (média móvel) para {symbol}")
            current = float(df['close'].iloc[-1])
            momentum = float((df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5])
            predicted = current * (1 + momentum * 0.3)
            
            return {
                'predicted_price': predicted,
                'model_type': '⚠️ Fallback (Média Móvel)'
            }
        
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        source = model_data['source']
        arch_type = model_data.get('model_type', 'unknown')
        
        # Fazer previsao
        try:
            X = preprocessor.transform_for_prediction(df)
            predictions = model.predict(X)
            pred_scaled = predictions[0]
            predicted_price = preprocessor.inverse_transform_target(pred_scaled)
            
            # Determinar nome do modelo para exibição (CORRIGIDO - sem "Fine-tuned")
            if source == "hub" or source == "local":
                if arch_type == 'improved':
                    model_type_display = f"🎯 LSTM Específico ({symbol})"
                else:
                    model_type_display = f"📊 LSTM ({symbol})"
            elif source == "base":
                model_type_display = "🧠 LSTM Base (genérico)"
            else:
                model_type_display = f"LSTM ({source})"
            
            logger.info(f"✅ Previsão concluída | {symbol} | ${predicted_price:.2f} | {model_type_display}")
            
            return {
                'predicted_price': float(predicted_price),
                'model_type': model_type_display
            }
        except Exception as e:
            logger.error(f"❌ Erro na previsão para {symbol}: {e}")
            current = float(df['close'].iloc[-1])
            return {
                'predicted_price': current,
                'model_type': f'⚠️ Erro: {str(e)[:30]}'
            }
    
    def list_available_models(self) -> List[str]:
        """Lista modelos disponiveis."""
        models = list(self.AVAILABLE_MODELS)
        
        # Adicionar modelos locais
        for f in self.LOCAL_CACHE.glob("lstm_model_*.pth"):
            name = f.stem.replace("lstm_model_", "")
            if name not in models:
                models.append(name)
        
        return models
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Limpa cache de modelos."""
        if symbol:
            symbol = symbol.upper()
            if symbol in self.model_cache:
                del self.model_cache[symbol]
                logger.info(f"🗑️ Cache limpo para {symbol}")
        else:
            self.model_cache.clear()
            logger.info("🗑️ Cache completo limpo")
    
    def get_model_info(self, symbol: str) -> dict:
        """
        Retorna informacoes detalhadas sobre o modelo em uso para um simbolo.
        Util para verificar qual modelo esta sendo usado.
        """
        symbol = symbol.upper()
        model_data = self.get_model(symbol)
        
        if not model_data:
            return {
                "symbol": symbol,
                "status": "not_loaded",
                "message": "Modelo não carregado ou não disponível"
            }
        
        model = model_data['model']
        source = model_data['source']
        arch_type = model_data.get('model_type', 'unknown')
        
        # Extrair informacoes do modelo
        info = {
            "symbol": symbol,
            "status": "loaded",
            "source": source,
            "architecture": arch_type,
            "in_cache": symbol in self.model_cache,
            "model_details": {}
        }
        
        # Detalhes especificos da arquitetura
        if hasattr(model, 'input_size'):
            info["model_details"]["input_size"] = model.input_size
        if hasattr(model, 'hidden_size'):
            info["model_details"]["hidden_size"] = model.hidden_size
        if hasattr(model, 'num_layers'):
            info["model_details"]["num_layers"] = model.num_layers
        if hasattr(model, 'dropout'):
            info["model_details"]["dropout"] = model.dropout
        if hasattr(model, 'bidirectional'):
            info["model_details"]["bidirectional"] = model.bidirectional
        
        # Descricao legivel
        if arch_type == 'improved':
            info["description"] = f"Modelo LSTM Bidirecional com Attention, treinado especificamente para {symbol}"
        else:
            info["description"] = f"Modelo LSTM original para {symbol}"
        
        if source == 'base':
            info["description"] = "Modelo BASE genérico (usado quando não há modelo específico)"
        
        logger.info(f"📋 Info do modelo {symbol}: {arch_type} from {source}")
        
        return info
    
    def get_all_loaded_models(self) -> List[dict]:
        """Retorna informacoes de todos os modelos em cache."""
        return [
            {
                "symbol": symbol,
                "source": data.get('source'),
                "architecture": data.get('model_type'),
                "symbol_requested": data.get('symbol_requested')
            }
            for symbol, data in self.model_cache.items()
        ]
