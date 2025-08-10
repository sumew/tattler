#!/bin/bash

# Setup script for Tattler - Drone Parking Lot Monitor
# This script sets up the development environment and validates the system

set -e  # Exit on any error

echo "🚁 Tattler - Development Environment Setup"
echo "=========================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() {
    echo -e "${BLUE}ℹ️  INFO${NC}: $1"
}

success() {
    echo -e "${GREEN}✅ SUCCESS${NC}: $1"
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

error() {
    echo -e "${RED}❌ ERROR${NC}: $1"
}

# Check prerequisites
echo "🔧 Checking Prerequisites..."
echo

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    success "Docker found: $DOCKER_VERSION"
else
    error "Docker is not installed. Please install Docker first."
    echo "  Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    success "Docker Compose found: $COMPOSE_VERSION"
else
    error "Docker Compose is not installed. Please install Docker Compose first."
    echo "  Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if Docker daemon is running
if docker info &> /dev/null; then
    success "Docker daemon is running"
else
    error "Docker daemon is not running. Please start Docker first."
    exit 1
fi

echo

# Validate project structure
echo "📁 Validating Project Structure..."
echo

required_dirs=("services" "videos" "scripts")
required_files=("docker-compose.yml" "README.md" "SYSTEM_DESIGN.md")

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        success "Directory exists: $dir"
    else
        warn "Directory missing: $dir (will be created)"
        mkdir -p "$dir"
        success "Created directory: $dir"
    fi
done

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        success "File exists: $file"
    else
        error "Required file missing: $file"
    fi
done

echo

# Check service directories
echo "🔍 Checking Service Directories..."
echo

services=("ffmpeg_extraction_job" "detection_orchestrator" "vehicle_detection_service" "persistence_service" "reporting")

for service in "${services[@]}"; do
    service_dir="services/$service"
    
    if [ -d "$service_dir" ]; then
        success "Service directory exists: $service"
        
        # Check for Dockerfile
        if [ -f "$service_dir/Dockerfile" ]; then
            success "  Dockerfile found"
        else
            warn "  Dockerfile missing"
        fi
        
        # Check for main application file
        if [ -f "$service_dir/main.py" ]; then
            success "  Main application file found"
        else
            warn "  Main application file missing"
        fi
        
        # Check for requirements.txt
        if [ -f "$service_dir/requirements.txt" ]; then
            success "  Requirements file found"
        else
            warn "  Requirements file missing"
        fi
    else
        error "Service directory missing: $service"
    fi
done

echo

# Validate Docker Compose configuration
echo "📋 Validating Docker Compose Configuration..."
echo

if docker-compose config &> /dev/null; then
    success "Docker Compose configuration is valid"
else
    error "Docker Compose configuration is invalid"
    info "Run 'docker-compose config' to see detailed errors"
fi

echo

# Build all services
echo "🔨 Building All Services..."
echo

info "This may take a few minutes on first run..."

if docker-compose build &> /dev/null; then
    success "All services built successfully"
else
    error "Service build failed"
    info "Run 'docker-compose build' to see detailed errors"
    exit 1
fi

echo

# Test core services
echo "🧪 Testing Core Services..."
echo

info "Starting Redis and PostgreSQL..."
docker-compose up -d redis postgres &> /dev/null
sleep 5

# Test Redis
if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
    success "Redis is working"
else
    error "Redis test failed"
fi

# Test PostgreSQL
if docker-compose exec -T postgres pg_isready -U user -d tattler_db | grep -q "accepting connections"; then
    success "PostgreSQL is working"
else
    error "PostgreSQL test failed"
fi

echo

# Test YOLO detector
echo "🤖 Testing YOLO Vehicle Detector..."
echo

info "This tests the core AI functionality..."

if docker-compose run --rm vehicle_detection_service python test_detector_simple.py &> /dev/null; then
    success "YOLO detector is working"
else
    error "YOLO detector test failed"
    info "Check service logs: docker-compose logs vehicle_detection_service"
fi

echo

# Create sample video if needed
echo "🎬 Setting Up Sample Data..."
echo

if [ ! -f "videos/cars.mp4" ]; then
    warn "No sample video found"
    info "Creating a test video for demonstration..."
    
    # Create a simple test video using ffmpeg if available
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -f lavfi -i testsrc=duration=10:size=320x240:rate=1 -pix_fmt yuv420p videos/cars.mp4 -y &> /dev/null
        success "Sample test video created: videos/cars.mp4"
    else
        warn "FFmpeg not available locally - sample video not created"
        info "You can add your own video files to the videos/ directory"
    fi
else
    success "Sample video exists: videos/cars.mp4"
fi

echo

# Make scripts executable
echo "🔐 Setting Script Permissions..."
echo

scripts=("process_video.sh" "quick_process.sh" "test_system.sh")

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        success "Made executable: $script"
    else
        warn "Script not found: $script"
    fi
done

echo

# Cleanup test services
info "Cleaning up test services..."
docker-compose down &> /dev/null

echo

# Final summary
echo "🎉 Environment Setup Complete!"
echo "=============================="
echo
echo "Your Tattler system is ready to use!"
echo
echo "Next steps:"
echo "  1. Add video files to the videos/ directory"
echo "  2. Run a full system test: ./test_system.sh"
echo "  3. Process a video: ./process_video.sh your_video.mp4"
echo "  4. Or use Make commands: make help"
echo
echo "Useful commands:"
echo "  make test-detector    # Test YOLO AI detector"
echo "  make status          # Check service status"
echo "  make help            # Show all available commands"
echo
echo "Documentation:"
echo "  README.md           # User guide"
echo "  SYSTEM_DESIGN.md    # Technical architecture"
echo "  CURRENT_STATUS.md   # System status and capabilities"
echo
success "Setup completed successfully!"
