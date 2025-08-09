# Tattler - Drone Parking Lot Monitor (Batch Processing)

This project uses a fleet of drones to monitor a parking lot. It processes video files in a multi-stage batch process using a service-oriented architecture.

See `SYSTEM_DESIGN.md` for a full overview of the architecture.

## Prerequisites

- Docker
- Docker Compose

## Architecture Improvements

This project now uses:
- **Service-specific requirements**: Each service has its own `requirements.txt` file with only the dependencies it needs
- **Docker volumes**: Shared data between containers uses Docker volumes for optimal performance
- **Network isolation**: All services communicate through a dedicated Docker network
- **Read-only volumes**: Services that only consume data have read-only access to prevent accidental modifications

## Quick Start (Automated)

### Option 1: Full Automated Processing
```bash
./process_video.sh your_video.mp4
```
This script runs the entire pipeline automatically and waits for completion.

### Option 2: Quick Start (Manual Control)
```bash
./quick_process.sh your_video.mp4
```
This starts the pipeline but lets you control the later stages manually.

## Manual Processing (Step by Step)

If you prefer to run each step manually:

### 1. Start Core Services
```bash
docker-compose up -d redis postgres persistence_service vehicle_detection_service
```

### 2. Place Video Files
Add your drone video files (e.g., `flight_001.mp4`) into the `videos/` directory.

### 3. Extract Frames
```bash
VIDEO_FILE=your_video.mp4 docker-compose run --rm ffmpeg_extraction_job
```

### 4. Queue Detection Jobs
```bash
docker-compose run --rm detection_orchestrator
```

### 5. Monitor Vehicle Detection Progress
```bash
docker-compose logs -f vehicle_detection_service
```
Wait for this to complete before proceeding.

### 6. Run License Plate Recognition
```bash
docker-compose run --rm lpr_job
```

### 7. Generate Reports
```bash
docker-compose run --rm reporting_engine
```

### 8. Stop Services
```bash
docker-compose down
```

## Video File Options

- **Single file**: `VIDEO_FILE=flight_001.mp4`
- **Multiple files**: `VIDEO_FILE="flight_001.mp4,flight_002.mp4"`
- **All MP4 files**: `VIDEO_FILE=*.mp4`
- **Default**: Uses `cars.mp4` if no file specified

## Volume Management

The system uses the following Docker volumes:
- `raw_frames_data`: Stores extracted frames from videos
- `detected_frames_data`: Stores frames with detected vehicles
- `redis_data`: Persistent Redis data
- `postgres_data`: Persistent PostgreSQL data

## Monitoring and Debugging

- **Check service logs**: `docker-compose logs [service_name]`
- **View volume contents**: `docker run --rm -v tattler_raw_frames_data:/data alpine ls /data`
- **Copy frames for inspection**: `docker run --rm -v tattler_raw_frames_data:/source -v $(pwd):/dest alpine cp /source/frame_name.jpg /dest/`

## Shutting Down

To stop all running services:
```bash
docker-compose down
```

To also remove volumes (WARNING: This will delete all processed data):
```bash
docker-compose down -v
```
