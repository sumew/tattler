# Tattler - Drone Parking Lot Monitor
# Makefile for common operations

.PHONY: help build up down logs clean test-detector test-plates process quick-process

# Default target
help:
	@echo "🚁 Tattler - Drone Parking Lot Monitor"
	@echo ""
	@echo "Available commands:"
	@echo "  make build          - Build all Docker services"
	@echo "  make up             - Start core services (Redis, PostgreSQL)"
	@echo "  make down           - Stop all services"
	@echo "  make logs           - View logs from all services"
	@echo "  make clean          - Stop services and remove volumes (⚠️  deletes data)"
	@echo "  make test-detector  - Test YOLO vehicle detector"
	@echo "  make test-plates    - Test license plate recognizer"
	@echo "  make process VIDEO=<file> - Run full automated pipeline"
	@echo "  make quick-process VIDEO=<file> - Quick setup for manual control"
	@echo ""
	@echo "Examples:"
	@echo "  make process VIDEO=flight_001.mp4"
	@echo "  make quick-process VIDEO=drone_footage.mp4"
	@echo ""
	@echo "Manual pipeline steps:"
	@echo "  make extract VIDEO=<file>     - Extract frames from video"
	@echo "  make queue                    - Queue frames for detection"
	@echo "  make detect                   - Run vehicle detection"
	@echo "  make detect-simple            - Run simplified vehicle detection"
	@echo "  make plates                   - Run license plate recognition"
	@echo "  make report                   - Generate reports"

# Build all services
build:
	@echo "🔨 Building all Docker services..."
	docker-compose build

# Start core services
up:
	@echo "🚀 Starting core services..."
	docker-compose up -d redis

# Stop all services
down:
	@echo "🛑 Stopping all services..."
	docker-compose down

# View logs
logs:
	@echo "📋 Viewing service logs..."
	docker-compose logs -f

# Clean everything (removes volumes)
clean:
	@echo "🧹 Cleaning up (this will delete all data)..."
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		docker system prune -f; \
	fi

# Test YOLO detector
test-detector:
	@echo "🧪 Testing YOLO vehicle detector..."
	docker-compose run --rm vehicle_detection_service python test_detector_simple.py

# Test license plate recognizer
test-plates:
	@echo "🧪 Testing license plate recognizer..."
	docker-compose run --rm plate_recognition_service python -m pytest tests/ -v

# Full automated processing
process:
	@if [ -z "$(VIDEO)" ]; then \
		echo "❌ Please specify VIDEO parameter: make process VIDEO=your_video.mp4"; \
		exit 1; \
	fi
	@echo "🎬 Running full automated pipeline for $(VIDEO)..."
	./process_video.sh $(VIDEO)

# Quick processing setup
quick-process:
	@if [ -z "$(VIDEO)" ]; then \
		echo "❌ Please specify VIDEO parameter: make quick-process VIDEO=your_video.mp4"; \
		exit 1; \
	fi
	@echo "⚡ Running quick setup for $(VIDEO)..."
	./quick_process.sh $(VIDEO)

# Manual pipeline steps
extract:
	@if [ -z "$(VIDEO)" ]; then \
		echo "❌ Please specify VIDEO parameter: make extract VIDEO=your_video.mp4"; \
		exit 1; \
	fi
	@echo "🎞️  Extracting frames from $(VIDEO)..."
	VIDEO_FILE=$(VIDEO) docker-compose run --rm ffmpeg_extraction_job

queue:
	@echo "📋 Frames are processed directly by vehicle detection service (no queuing needed)"

detect:
	@echo "🚗 Running vehicle detection service..."
	docker-compose up -d vehicle_detection_service
	@echo "Monitor with: docker-compose logs -f vehicle_detection_service"

detect-simple:
	@echo "🚗 Running simplified vehicle detection..."
	docker-compose run --rm vehicle_detection_service python main_simple.py

plates:
	@echo "🔤 Running license plate recognition..."
	docker-compose run --rm plate_recognition_service

report:
	@echo "📊 Reporting service removed - query PostgreSQL database directly"
	@echo "   Connect: docker-compose exec postgres psql -U user -d tattler_db"

# Debugging and inspection commands
inspect-frames:
	@echo "🔍 Inspecting extracted frames..."
	docker run --rm -v tattler_raw_frames_data:/data alpine ls -la /data

inspect-detected:
	@echo "🔍 Inspecting detected frames..."
	docker run --rm -v tattler_detected_frames_data:/data alpine ls -la /data

inspect-vehicle-crops:
	@echo "🔍 Inspecting vehicle crops..."
	docker run --rm -v tattler_vehicle_crops_data:/data alpine ls -la /data

inspect-plate-crops:
	@echo "🔍 Inspecting license plate crops..."
	docker run --rm -v tattler_plate_crops_data:/data alpine ls -la /data

status:
	@echo "📊 Service status:"
	docker-compose ps

# Development commands
rebuild:
	@echo "🔄 Rebuilding all services..."
	docker-compose build --no-cache

rebuild-detector:
	@echo "🔄 Rebuilding vehicle detection service..."
	docker-compose build --no-cache vehicle_detection_service
