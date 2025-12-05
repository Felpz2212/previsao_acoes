"""
Configuration settings for the Stock Prediction LSTM project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", None)

# Stock Configuration
DEFAULT_STOCK_SYMBOL = os.getenv("DEFAULT_STOCK_SYMBOL", "AAPL")
DEFAULT_START_DATE = os.getenv("DEFAULT_START_DATE", "2018-01-01")
DEFAULT_END_DATE = os.getenv("DEFAULT_END_DATE", "2024-12-31")

# LSTM Model Configuration
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", "60"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "50"))
LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "32"))
LSTM_LEARNING_RATE = float(os.getenv("LSTM_LEARNING_RATE", "0.001"))
LSTM_HIDDEN_SIZE = 50
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# Monitoring
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Model paths
def get_model_path(symbol: str) -> Path:
    """Get the path for a model file based on stock symbol."""
    return MODELS_DIR / f"lstm_model_{symbol}.pth"

def get_scaler_path(symbol: str) -> Path:
    """Get the path for a scaler file based on stock symbol."""
    return MODELS_DIR / f"scaler_{symbol}.pkl"

