#!/bin/bash

# Smart Mirror Auto-start Script
echo "🚀 Starting Smart Mirror Dermatologist Application..."

# Wait for desktop environment to load (important for auto-boot)
sleep 5

# Set display for GUI applications
export DISPLAY=:0

# Navigate to project directory
cd ~/smart-mirror-project

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️ Virtual environment not found. Running installation..."
    ./install_standalone.sh
fi

# Activate virtual environment
source venv/bin/activate

# Create log directory if it doesn't exist
mkdir -p logs

# Start application with comprehensive logging
echo "📹 Launching Smart Mirror Dermatologist..."
python3 smart_mirror_standalone.py 2>&1 | tee logs/smart_mirror_$(date +%Y%m%d_%H%M%S).log