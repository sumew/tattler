#!/bin/bash

# Setup script for creating virtual environments for all services

set -e  # Exit on any error

echo "🚀 Setting up virtual environments for all services..."
echo

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo

# Services to set up
services=("vehicle_detection" "license_plate_recognition" "frame_extraction" "persistence")

for service in "${services[@]}"; do
    echo "📦 Setting up $service..."
    
    service_dir="services_new/$service"
    
    if [ ! -d "$service_dir" ]; then
        echo "❌ Service directory not found: $service_dir"
        continue
    fi
    
    cd "$service_dir"
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        echo "  Creating virtual environment..."
        python3 -m venv venv
    else
        echo "  Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    echo "  Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install requirements if they exist
    if [ -f "requirements.txt" ]; then
        echo "  Installing requirements..."
        pip install -r requirements.txt --quiet
        echo "  ✅ Requirements installed"
    else
        echo "  ⚠️  No requirements.txt found"
    fi
    
    # Deactivate virtual environment
    deactivate
    
    cd ../..
    echo "  ✅ $service setup complete"
    echo
done

echo "🎉 All virtual environments set up successfully!"
echo
echo "To test individual services:"
echo "  cd services_new/vehicle_detection && source venv/bin/activate && python test_standalone.py"
echo "  cd services_new/license_plate_recognition && source venv/bin/activate && python test_standalone.py"
echo
echo "Or use the Makefile:"
echo "  make test-vehicle"
echo "  make test-lpr"
