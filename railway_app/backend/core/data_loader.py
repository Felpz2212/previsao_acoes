"""
Data loading module for stock price data from Yahoo Finance.
Versao robusta com multiplas tentativas e fallbacks.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
from loguru import logger
import time


class StockDataLoader:
    """Load and manage stock price data from Yahoo Finance."""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # segundos
    
    def __init__(self):
        """Initialize the data loader."""
        pass
    
    def _download_with_retry(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Tenta baixar dados com multiplas estrategias."""
        
        # Estrategia 1: yf.download()
        for attempt in range(self.MAX_RETRIES):
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True,
                    timeout=10
                )
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Tentativa {attempt+1} yf.download falhou: {e}")
                time.sleep(self.RETRY_DELAY)
        
        # Estrategia 2: yf.Ticker().history()
        for attempt in range(self.MAX_RETRIES):
            try:
                ticker = yf.Ticker(symbol)
                # Calcular periodo em dias
                start_dt = datetime.strptime(start, '%Y-%m-%d')
                end_dt = datetime.strptime(end, '%Y-%m-%d')
                days = (end_dt - start_dt).days
                
                # Usar periodo relativo
                period = "1y" if days <= 365 else "2y" if days <= 730 else "5y"
                
                df = ticker.history(period=period, auto_adjust=True)
                if not df.empty:
                    # Filtrar pelo periodo solicitado
                    df = df.loc[start:end]
                    return df
            except Exception as e:
                logger.warning(f"Tentativa {attempt+1} Ticker.history falhou: {e}")
                time.sleep(self.RETRY_DELAY)
        
        # Estrategia 3: Tentar com periodo fixo
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="max", auto_adjust=True)
            if not df.empty:
                return df.tail(365)  # Ultimos 365 dias
        except Exception as e:
            logger.error(f"Todas as estrategias falharam para {symbol}: {e}")
        
        return pd.DataFrame()
    
    def load_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance com fallbacks.
        """
        try:
            logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
            
            df = self._download_with_retry(symbol, start_date, end_date)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Flatten column names if multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Handle different column names
            date_col = None
            for col in ['Date', 'date', 'Datetime', 'datetime', 'index']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df = df.rename(columns={date_col: 'timestamp'})
            elif 'timestamp' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'timestamp'})
            
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Remove timezone if present
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            # Add date components
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully loaded {len(df)} records for {symbol}")
            
            if save_path:
                df.to_csv(save_path, index=False)
                logger.info(f"Data saved to {save_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the loaded data."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df[required_columns].isnull().any().any():
            logger.warning("Found missing values in data")
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise ValueError("timestamp column must be datetime type")
        
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] < 0).any().any():
            raise ValueError("Found negative prices in data")
        
        if (df['high'] < df['low']).any():
            raise ValueError("Found records where high < low")
        
        logger.info("Data validation passed")
        return True
    
    def get_latest_price(self, symbol: str) -> dict:
        """Get the latest price information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'previous_close': info.get('previousClose'),
                'open': info.get('open', info.get('regularMarketOpen')),
                'day_high': info.get('dayHigh', info.get('regularMarketDayHigh')),
                'day_low': info.get('dayLow', info.get('regularMarketDayLow')),
                'volume': info.get('volume', info.get('regularMarketVolume')),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            raise


# Removido bloco de teste para deploy
