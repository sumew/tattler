#!/bin/bash

# Quick Tattler Processing (without waiting for completion)
# Usage: ./quick_process.sh [video_filename]

set -e

VIDEO_FILE=${1:-cars.mp4}
echo "🎬 Quick processing: $VIDEO_FILE"

# Start services
echo "📡 Starting services..."
docker-compose up -d redis postgres persistence_service vehicle_detection_service

# Extract frames
echo "🎞️  Extracting frames..."
VIDEO_FILE=$VIDEO_FILE docker-compose run --rm ffmpeg_extraction_job

# Queue detection jobs
echo "🔍 Queuing detection jobs..."
docker-compose run --rm detection_orchestrator

echo "✅ Jobs queued! Vehicle detection is running in the background."
echo "📊 Check progress with: docker-compose logs vehicle_detection_service"
echo "🔄 Run LPR when ready: docker-compose run --rm lpr_job"
echo "📋 Generate reports: docker-compose run --rm reporting_engine"
echo "🛑 Stop services: docker-compose down"
