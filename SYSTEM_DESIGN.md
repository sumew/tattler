# Tattler Drone Parking Lot Monitoring System (Batch Processing)

## 1. Overview

Tattler is a distributed system designed to monitor parking lots using a fleet of drones. It processes recorded video files to identify vehicle license plates, record their GPS locations, and provide analytics on parking duration. The system is built on a scalable, service-oriented architecture where long-running services listen for work on a message bus.

## 2. System Architecture

The architecture is composed of on-demand jobs and persistent services that communicate via Redis.

**High-Level Data Flow:**

```
[Video File] -> [Job 1: FFmpeg Extraction] -> [Raw Frames]
                                                     |
                                                     v
[Job 2: Detection Orchestrator] -> [Redis Channel: frames_to_detect] -> [Service: Vehicle Detection] -> [Detected Frames]
                                                                                                           |
                                                                                                           v
                                                                         [Job 3: LPR Processor] -> [Redis Channel: plate_sightings] -> [Service: Persistence] -> [PostgreSQL DB]
```

### 2.1. Components

#### a. Storage Directories
- **`videos/`:** Raw video files from drones.
- **`raw_frames/`:** All frames extracted from a video by FFmpeg.
- **`detected_frames/`:** Frames confirmed to contain a vehicle and a license plate.

#### b. On-Demand Jobs
- **`ffmpeg_extraction_job`:** Takes a video file as input and rapidly extracts all its frames into the `raw_frames/` directory.
- **`detection_orchestrator`:** Scans the `raw_frames/` directory and publishes a message for each frame to the `frames_to_detect` Redis channel. This queues up the work for the detection service.
- **`lpr_job`:** Scans the `detected_frames/` directory, runs OCR on each image, and publishes the results to the `plate_sightings` channel.

#### c. Long-Running Services
- **`vehicle_detection_service`:**
    - On startup, loads the vehicle and license plate detection models into memory.
    - Subscribes to the `frames_to_detect` Redis channel.
    - When it receives a message containing an image path, it processes that single file using its pre-loaded models.
    - If the image is valid, it moves it to `detected_frames/`; otherwise, it's deleted.
- **`persistence_service`:** Subscribes to the `plate_sightings` channel and saves the final results to the PostgreSQL database.

#### d. Message Bus (Redis)
- **`frames_to_detect` channel:** Carries messages with image paths, instructing the detection service what to process.
- **`plate_sightings` channel:** Carries the final results (plate number, location) to be saved.