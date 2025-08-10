#!/bin/bash

# Tattler - Python Environment Setup
# Sets up virtual environments and installs Python dependencies for each service

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo "🐍 Setting up Python environments for Tattler services..."
echo "======================================================="

# Main project virtual environment
log_info "Setting up main project environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_info "Created main venv"
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
log_success "Main project environment ready"

# Vehicle Detection Service
log_info "Setting up vehicle detection service environment..."
cd services/vehicle_detection_service
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_info "Created vehicle detection venv"
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ../..
log_success "Vehicle detection service environment ready"

# Plate Recognition Service
log_info "Setting up plate recognition service environment..."
cd services/plate_recognition
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_info "Created plate recognition service venv"
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ../..
log_success "Plate recognition service environment ready"

# FFmpeg Extraction (no Python deps needed)
log_info "FFmpeg extraction service uses system ffmpeg - no Python deps needed"

echo ""
log_success "🎉 All Python environments set up successfully!"
echo ""
echo "Virtual environments created:"
echo "  • Main project: ./venv"
echo "  • Vehicle detection: ./services/vehicle_detection_service/venv"
echo "  • Plate recognition: ./services/plate_recognition/venv"
echo ""
echo "To activate an environment:"
echo "  source venv/bin/activate  # Main project"
echo "  cd services/vehicle_detection_service && source venv/bin/activate  # Vehicle detection"
echo "  cd services/plate_recognition && source venv/bin/activate  # Plate recognition"
echo "  # etc..."
