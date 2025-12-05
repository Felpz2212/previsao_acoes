#!/usr/bin/env python3
"""
Script to train a stock prediction model.
Can be run as a scheduled job or manually.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import train_stock_model
from config.settings import DEFAULT_START_DATE, DEFAULT_END_DATE
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description='Train stock prediction model')
    parser.add_argument('symbol', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE,
                       help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=DEFAULT_END_DATE,
                       help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    logger.info(f"Training model for {args.symbol}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    try:
        metrics = train_stock_model(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {metrics}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

