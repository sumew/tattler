# Tattler - Drone Parking Lot Monitor (Restructured)

A modular, testable system for processing drone footage to detect vehicles and license plates.

## Project Structure

```
tattler/
├── services/
│   ├── vehicle_detection/          # YOLO-based vehicle detection
│   │   ├── venv/                   # Virtual environment
│   │   ├── requirements.txt
│   │   ├── main.py                 # Service entry point
│   │   ├── detector.py             # Core detection logic
│   │   ├── test_standalone.py      # Standalone testing
│   │   └── Dockerfile
│   ├── license_plate_recognition/  # OCR-based license plate recognition
│   │   ├── venv/
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   ├── ocr.py
│   │   ├── test_standalone.py
│   │   └── Dockerfile
│   ├── frame_extraction/           # Video to frames conversion
│   │   ├── venv/
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   ├── extractor.py
│   │   ├── test_standalone.py
│   │   └── Dockerfile
│   └── persistence/                # Database operations
│       ├── venv/
│       ├── requirements.txt
│       ├── main.py
│       ├── database.py
│       ├── test_standalone.py
│       └── Dockerfile
├── shared/
│   ├── __init__.py
│   ├── redis_client.py             # Shared Redis utilities
│   ├── config.py                   # Shared configuration
│   └── utils.py                    # Common utilities
├── tests/
│   ├── integration/                # End-to-end tests
│   ├── sample_images/              # Test images
│   └── sample_videos/              # Test videos
├── scripts/
│   ├── setup_environments.sh       # Set up all virtual environments
│   ├── test_all_services.sh        # Test all services individually
│   └── run_pipeline.sh             # Run full pipeline
├── docker-compose.yml              # For integrated testing
├── requirements-dev.txt            # Development dependencies
└── Makefile                        # Common commands
```

## Quick Start

### 1. Set up all virtual environments
```bash
make setup
```

### 2. Test individual services
```bash
# Test vehicle detection with a single image
cd services/vehicle_detection
python test_standalone.py path/to/image.jpg

# Test license plate recognition
cd services/license_plate_recognition  
python test_standalone.py path/to/image.jpg

# Test frame extraction
cd services/frame_extraction
python test_standalone.py path/to/video.mp4
```

### 3. Run full pipeline
```bash
make run-pipeline VIDEO=cars.mp4
```

## Individual Service Testing

Each service can be tested independently:

### Vehicle Detection
```bash
cd services/vehicle_detection
source venv/bin/activate
python test_standalone.py ../../../detected_frames_local/cars_frame_00001.jpg
```

### License Plate Recognition
```bash
cd services/license_plate_recognition
source venv/bin/activate
python test_standalone.py ../../../detected_frames_local/cars_frame_00001.jpg
```

## Development

### Adding a new service
1. Create service directory under `services/`
2. Run `python -m venv venv` in the service directory
3. Create `requirements.txt`, `main.py`, `test_standalone.py`
4. Add service to `docker-compose.yml`

### Running tests
```bash
make test-all          # Test all services
make test-vehicle      # Test vehicle detection only
make test-lpr          # Test license plate recognition only
```
