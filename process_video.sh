#!/bin/bash

# Tattler Video Processing Pipeline
# Usage: ./process_video.sh [video_filename]

set -e  # Exit on any error

VIDEO_FILE=${1:-cars.mp4}
echo "🎬 Processing video: $VIDEO_FILE"

# Check if video file exists
if [ ! -f "./videos/$VIDEO_FILE" ]; then
    echo "❌ Error: Video file './videos/$VIDEO_FILE' not found"
    exit 1
fi

echo "🚀 Starting Tattler video processing pipeline..."

# Step 1: Start core services
echo "📡 Starting core services (Redis, Postgres, Persistence, Vehicle Detection)..."
docker-compose up -d redis postgres persistence_service vehicle_detection_service

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Step 2: Extract frames
echo "🎞️  Extracting frames from $VIDEO_FILE..."
VIDEO_FILE=$VIDEO_FILE docker-compose run --rm ffmpeg_extraction_job

# Step 3: Queue detection jobs
echo "🔍 Queuing vehicle detection jobs..."
docker-compose run --rm detection_orchestrator

# Step 4: Wait for vehicle detection to complete
echo "🚗 Processing vehicle detection (this may take a while)..."
echo "   Monitoring detection progress..."

# Simple progress monitoring
TOTAL_FRAMES=$(docker run --rm -v tattler_raw_frames_data:/data alpine sh -c "ls /data | wc -l")
echo "   Total frames to process: $TOTAL_FRAMES"

# Wait and check progress periodically
while true; do
    DETECTED_FRAMES=$(docker run --rm -v tattler_detected_frames_data:/data alpine sh -c "ls /data 2>/dev/null | wc -l" || echo "0")
    echo "   Processed: $DETECTED_FRAMES/$TOTAL_FRAMES frames"
    
    if [ "$DETECTED_FRAMES" -eq "$TOTAL_FRAMES" ]; then
        echo "✅ Vehicle detection complete!"
        break
    fi
    
    sleep 30
done

# Step 5: Run license plate recognition
echo "🔤 Running license plate recognition..."
docker-compose run --rm lpr_job

# Step 6: Generate reports
echo "📊 Generating reports..."
docker-compose run --rm reporting_engine

echo "🎉 Video processing complete!"
echo "📋 Results have been stored in the database."
echo "🛑 To stop all services, run: docker-compose down"
