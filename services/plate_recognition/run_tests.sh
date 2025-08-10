#!/bin/bash
"""
Test runner for License Plate Recognition Service
"""

echo "🔤 Running License Plate Recognition Service Tests..."
echo "=" * 50

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Run the tests
python tests/test_plate_recognition.py
