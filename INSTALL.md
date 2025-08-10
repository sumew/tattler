# Tattler Installation Guide

## Development Setup

### 1. Install Root Package (Shared Utilities)
```bash
# From the repo root
pip install -e .
```

### 2. Install Services
```bash
# Vehicle Detection Service
pip install -e ./services/vehicle_detection

# Plate Recognition Service  
pip install -e ./services/plate_recognition

# FFmpeg Extraction Service
pip install -e ./services/ffmpeg_extraction_job
```

### 3. Verify Installation
```bash
python -c "from tattler.utils import load_image_from_input; print('✅ Shared utilities OK')"
python -c "from tattler_vehicle_detection import VehicleDetector; print('✅ Vehicle detection OK')"
python -c "from tattler_plate_recognition import PlateRecognizer; print('✅ Plate recognition OK')"
```

## Docker Deployment

### Build Services
```bash
# Build from repo root
docker build -f services/vehicle_detection/Dockerfile -t vehicle-detector .
docker build -f services/plate_recognition/Dockerfile -t plate-recognizer .
docker build -f services/ffmpeg_extraction_job/Dockerfile -t ffmpeg-extractor .
```

## Usage Examples

### Import Shared Utilities
```python
from tattler.utils.input_utils import load_image_from_input
```

### Import Service Classes
```python
from tattler_vehicle_detection.detector import VehicleDetector
from tattler_plate_recognition.recognizer import PlateRecognizer
```

### Run Tests
```python
# Vehicle detection test
cd services/vehicle_detection
python tests/test_extract_vehicles.py
```
