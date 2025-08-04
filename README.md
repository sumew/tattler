# Tattler - Drone Parking Lot Monitor (Batch Processing)

This project uses a fleet of drones to monitor a parking lot. It processes video files in a multi-stage batch process using a service-oriented architecture.

See `SYSTEM_DESIGN.md` for a full overview of the architecture.

## Prerequisites

- Docker
- Docker Compose

## Setup & Running

1.  **Start Core Services**
    - This command starts Redis, Postgres, the `persistence_service`, and the new long-running `vehicle_detection_service`.
    - The detection service will start and immediately begin listening for jobs on a Redis channel.
    ```bash
    docker-compose up -d --build redis postgres persistence_service vehicle_detection_service
    ```

2.  **Place Video Files**
    - Add your drone video files (e.g., `flight_001.mp4`) into the `videos/` directory.

3.  **Run the FFmpeg Extraction Job**
    - This will process a single specified video and dump all its frames into `raw_frames/`.
    - **Replace `flight_001.mp4` with your video file's name.**
    ```bash
    docker-compose run --rm ffmpeg_extraction_job flight_001.mp4
    ```

4.  **Run the Detection Orchestrator**
    - This job scans `raw_frames/` and sends a message to the `vehicle_detection_service` for each frame, telling it to start processing.
    ```bash
    docker-compose run --rm detection_orchestrator
    ```

5.  **Run the LPR Job**
    - After the detection service has processed the frames, this job will process the images in `detected_frames/`.
    ```bash
    docker-compose run --rm lpr_job
    ```

6.  **Run the Reporting Engine**
    - To manually generate a report from the data in the database:
    ```bash
    docker-compose run --rm reporting_engine
    ```

## Shutting Down

To stop all running services:
```bash
docker-compose down
```
