#!/bin/bash

# Setup cron job for scheduled training
# Run this script to install the cron job

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create cron entry
# This will run training every Sunday at 2 AM
CRON_ENTRY="0 2 * * 0 $PROJECT_DIR/scripts/scheduled_training.sh"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "scheduled_training.sh"; then
    echo "Cron job already exists"
else
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo "Cron job installed successfully"
    echo "Model retraining will run every Sunday at 2 AM"
fi

# Display current crontab
echo ""
echo "Current crontab:"
crontab -l

