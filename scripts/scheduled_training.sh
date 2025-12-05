#!/bin/bash

# Scheduled training script
# This script can be run as a cron job to retrain models periodically

# Set project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Log file
LOG_FILE="logs/scheduled_training_$(date +%Y%m%d).log"

echo "$(date): Starting scheduled training" >> "$LOG_FILE"

# List of stocks to train
STOCKS=("AAPL" "GOOGL" "MSFT" "AMZN")

# Train models for each stock
for SYMBOL in "${STOCKS[@]}"; do
    echo "$(date): Training $SYMBOL" >> "$LOG_FILE"
    python scripts/train_model.py "$SYMBOL" --start-date 2020-01-01 --end-date $(date +%Y-%m-%d) 2>&1 | tee -a "$LOG_FILE"
done

echo "$(date): Scheduled training completed" >> "$LOG_FILE"

# Deactivate virtual environment
if [ -d "venv" ]; then
    deactivate
fi

