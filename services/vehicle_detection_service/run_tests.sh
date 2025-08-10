#!/bin/bash

# Vehicle Detection Service Test Runner

echo "🚗 Running Vehicle Detection Service Tests..."
echo "=============================================="

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export ULTRALYTICS_OFFLINE=1
export YOLO_CONFIG_DIR=/tmp

# Run the tests
python tests/test_vehicle_detection.py

echo ""
echo "📁 Check results in: test_data/images/results/"
echo "🎯 Test complete!"
