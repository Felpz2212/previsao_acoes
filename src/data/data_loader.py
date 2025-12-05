"""
Data loading module for stock price data from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
from loguru import logger


class StockDataLoader:
    """Load and manage stock price data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the data loader."""
        pass
    
    def load_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            save_path: Optional path to save the downloaded data
            
        Returns:
            DataFrame with stock price data
        """
        try:
            logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
            
            # Download data from Yahoo Finance
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True  # Get adjusted prices
            )
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Flatten column names if multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Handle different column names from yfinance versions
            date_col = None
            for col in ['Date', 'date', 'Datetime', 'datetime', 'index']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df = df.rename(columns={date_col: 'timestamp'})
            elif 'timestamp' not in df.columns:
                # If no date column found, use the first column
                df = df.rename(columns={df.columns[0]: 'timestamp'})
            
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add date components for potential feature engineering
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully loaded {len(df)} records for {symbol}")
            
            # Save to file if path provided
            if save_path:
                df.to_csv(save_path, index=False)
                logger.info(f"Data saved to {save_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded data for completeness and correctness.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes, raises exception otherwise
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required columns exist
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values in critical columns
        if df[required_columns].isnull().any().any():
            logger.warning("Found missing values in data")
            # Log which columns have missing values
            null_counts = df[required_columns].isnull().sum()
            logger.warning(f"Null counts: {null_counts[null_counts > 0]}")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise ValueError("timestamp column must be datetime type")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] < 0).any().any():
            raise ValueError("Found negative prices in data")
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            raise ValueError("Found records where high < low")
        
        logger.info("Data validation passed")
        return True
    
    def get_latest_price(self, symbol: str) -> dict:
        """
        Get the latest price information for a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with latest price information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', None),
                'previous_close': info.get('previousClose', None),
                'open': info.get('open', None),
                'day_high': info.get('dayHigh', None),
                'day_low': info.get('dayLow', None),
                'volume': info.get('volume', None),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            raise


# Backward compatibility with existing code
def load_actions(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with stock price data
    """
    loader = StockDataLoader()
    df = loader.load_stock_data(symbol, start_date, end_date)
    
    # Keep original column format for compatibility
    df = df.rename(columns={'timestamp': 'date'})
    df['ano'] = df['year']
    df['mes'] = df['month']
    df['dia'] = df['day']
    
    return df


if __name__ == "__main__":
    # Test the data loader
    loader = StockDataLoader()
    df = loader.load_stock_data("AAPL", "2018-01-01", "2024-12-31")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

