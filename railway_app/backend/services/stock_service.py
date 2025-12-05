"""
Stock Service - Dados de acoes com persistencia em PostgreSQL
"""
from typing import Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import time
from loguru import logger

# Importar de core/ (copia local para deploy)
from core.data_loader import StockDataLoader

# Database (opcional)
try:
    from database.service import get_db_service, DatabaseService
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("⚠️ Database não disponível, usando apenas cache em memória")


class StockService:
    """Servico para obter dados de acoes com persistencia."""
    
    _cache: Dict[str, dict] = {}
    _cache_ttl = 300  # 5 minutos
    
    COMPANY_NAMES = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla Inc.",
        "NFLX": "Netflix Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America Corp.",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc.",
        "WMT": "Walmart Inc.",
        "KO": "The Coca-Cola Company",
        "DIS": "The Walt Disney Company",
        "JNJ": "Johnson & Johnson",
        "PFE": "Pfizer Inc.",
        "PETR4.SA": "Petrobras",
        "VALE3.SA": "Vale S.A.",
        "ITUB4.SA": "Itau Unibanco",
        "BBDC4.SA": "Bradesco",
        "ABEV3.SA": "Ambev S.A.",
        "NU": "Nubank",
        "MELI": "MercadoLibre Inc.",
    }
    
    def __init__(self):
        self.loader = StockDataLoader()
        self.db: Optional[DatabaseService] = None
        
        if DB_AVAILABLE:
            try:
                self.db = get_db_service()
                logger.info("✅ StockService conectado ao PostgreSQL")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao conectar DB: {e}")
                self.db = None
    
    def _get_cache_key(self, symbol: str, days: int) -> str:
        return f"{symbol}_{days}"
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cached = self._cache[key]
        return (time.time() - cached['timestamp']) < self._cache_ttl
    
    def get_stock_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Obtem dados historicos com a seguinte prioridade:
        1. Cache em memoria (5 min)
        2. Banco de dados PostgreSQL
        3. Yahoo Finance API (e salva no DB)
        """
        symbol = symbol.upper()
        cache_key = self._get_cache_key(symbol, days)
        
        # 1. Verificar cache em memória
        if self._is_cache_valid(cache_key):
            logger.info(f"⚡ Cache hit para {symbol}")
            return self._cache[cache_key]['data'].copy()
        
        # 2. Tentar banco de dados
        if self.db:
            try:
                start_date = datetime.now() - timedelta(days=days + 30)
                df = self.db.get_stock_prices(symbol, start_date=start_date)
                
                if not df.empty and len(df) >= days * 0.8:  # 80% dos dados
                    logger.info(f"🗃️ Dados carregados do PostgreSQL para {symbol}")
                    
                    # Atualizar cache
                    self._cache[cache_key] = {
                        'data': df,
                        'timestamp': time.time()
                    }
                    return df.tail(days).copy()
                    
            except Exception as e:
                logger.warning(f"⚠️ Erro ao buscar do DB: {e}")
        
        # 3. Baixar do Yahoo Finance
        logger.info(f"🌐 Baixando dados do Yahoo Finance para {symbol}")
        
        end = datetime.now()
        start = end - timedelta(days=days + 30)
        
        try:
            df = self.loader.load_stock_data(
                symbol,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d')
            )
            
            if df.empty:
                return None
            
            self.loader.validate_data(df)
            
            # Salvar no banco de dados
            if self.db:
                try:
                    self.db.save_stock_prices(symbol, df)
                    logger.info(f"💾 Dados salvos no PostgreSQL para {symbol}")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao salvar no DB: {e}")
            
            # Atualizar cache
            df_limited = df.tail(days)
            self._cache[cache_key] = {
                'data': df_limited,
                'timestamp': time.time()
            }
            
            return df_limited.copy()
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter dados de {symbol}: {e}")
            return None
    
    def get_company_name(self, symbol: str) -> str:
        """Obtem nome da empresa pelo ticker."""
        return self.COMPANY_NAMES.get(symbol.upper(), symbol)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtem preco atual."""
        df = self.get_stock_data(symbol, days=5)
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[-1])
        return None
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Limpa cache."""
        if symbol:
            keys_to_remove = [k for k in self._cache if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
    
    def get_data_stats(self) -> Dict:
        """Retorna estatísticas dos dados armazenados."""
        stats = {
            'cache_entries': len(self._cache),
            'db_available': self.db is not None
        }
        
        if self.db:
            try:
                # Contar registros no DB
                from database.models import StockPrice
                session = self.db.get_session()
                stats['db_records'] = session.query(StockPrice).count()
                stats['db_symbols'] = session.query(StockPrice.symbol).distinct().count()
                session.close()
            except:
                pass
        
        return stats
