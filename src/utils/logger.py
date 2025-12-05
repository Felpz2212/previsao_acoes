"""
Logging configuration and utilities.
"""
import sys
from loguru import logger
from pathlib import Path
from config.settings import LOGS_DIR, LOG_LEVEL


def setup_logging():
    """Configure application logging."""
    
    # Remove default logger
    logger.remove()
    
    # Add console logger with colored output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        colorize=True
    )
    
    # Add file logger for all logs
    logger.add(
        LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=LOG_LEVEL
    )
    
    # Add error log file
    logger.add(
        LOGS_DIR / "errors_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="90 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR"
    )
    
    logger.info("Logging configured successfully")


# Setup logging when module is imported
setup_logging()

