# Tattler - Drone Parking Lot Monitor

AI-powered parking lot monitoring system using drone footage. Tattler automatically detects vehicles and license plates in video streams, providing comprehensive parking lot surveillance and analytics.

## 🚀 Features

- **Vehicle Detection**: AI-powered detection of cars, motorcycles, buses, and trucks
- **License Plate Recognition**: High-accuracy license plate detection and extraction
- **Video Processing**: Efficient frame extraction from drone footage
- **Modular Architecture**: Independent microservices for scalable deployment
- **Docker Ready**: Containerized services for easy deployment
- **Real-time Processing**: Optimized for live video stream analysis

## 🏗️ Architecture

Tattler is built as a collection of independent microservices:

```
tattler/
├── src/tattler/                    # Shared utilities
│   └── utils/
├── services/
│   ├── vehicle_detection/          # YOLO-based vehicle detection
│   ├── plate_recognition/          # License plate detection
│   └── ffmpeg_extraction_job/      # Video frame extraction
└── scripts/                        # Deployment and setup scripts
```

## 📦 Services

### 🚗 Vehicle Detection Service
AI-powered vehicle detection using YOLOv8 deep learning model.

**Features:**
- Detects cars, motorcycles, buses, and trucks
- Extracts individual vehicle images
- Creates annotated visualizations
- High-speed inference (~50ms per image)

**Quick Start:**
```python
from tattler_vehicle_detection.detector import VehicleDetector

detector = VehicleDetector()
detections = detector.detect_vehicles("parking_lot.jpg")
vehicle_crops = detector.extract_vehicle_images("parking_lot.jpg", detections)
```

[📖 Full Documentation](services/vehicle_detection/README.md)

### 🔢 Plate Recognition Service
Specialized license plate detection using YOLO-v9 model.

**Features:**
- High-accuracy license plate detection
- Plate annotation and extraction
- Optimized for various lighting conditions
- Simple, focused API

**Quick Start:**
```python
from tattler_plate_recognition.recognizer import PlateRecognizer

recognizer = PlateRecognizer()
detections = recognizer.detect_plate("car_image.jpg")
plates = recognizer.extract_plate("car_image.jpg", detections)
```

[📖 Full Documentation](services/plate_recognition/README.md)

### 🎬 FFmpeg Extraction Service
High-performance video frame extraction using FFmpeg.

**Features:**
- Extract frames at custom intervals
- Support for all major video formats
- Quality and resolution control
- Memory-efficient streaming processing

**Quick Start:**
```python
from tattler_ffmpeg.main import extract_frames

result = extract_frames(
    input_video="drone_footage.mp4",
    output_dir="frames/",
    interval=5.0
)
```

[📖 Full Documentation](services/ffmpeg_extraction_job/README.md)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for video processing)
- CUDA-compatible GPU (optional, for faster inference)

### Quick Install
```bash
# Clone the repository
git clone <repository-url>
cd tattler

# Install shared utilities
pip install -e .

# Install services
pip install -e ./services/vehicle_detection
pip install -e ./services/plate_recognition
pip install -e ./services/ffmpeg_extraction_job

# Install FFmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

[📖 Detailed Installation Guide](INSTALL.md)

## 🚀 Quick Start

### Complete Processing Pipeline
```python
from tattler_ffmpeg.main import extract_frames
from tattler_vehicle_detection.detector import VehicleDetector
from tattler_plate_recognition.recognizer import PlateRecognizer

# Step 1: Extract frames from drone video
frames = extract_frames("drone_footage.mp4", "frames/", interval=5.0)

# Step 2: Detect vehicles in each frame
vehicle_detector = VehicleDetector()
plate_recognizer = PlateRecognizer()

for frame_file in frames['output_files']:
    frame_path = f"frames/{frame_file}"
    
    # Detect vehicles
    vehicle_detections = vehicle_detector.detect_vehicles(frame_path)
    
    if vehicle_detections['has_vehicles']:
        # Extract vehicle images
        vehicles = vehicle_detector.extract_vehicle_images(frame_path, vehicle_detections)
        
        # Detect license plates in each vehicle
        for i, vehicle_img in enumerate(vehicles):
            plate_detections = plate_recognizer.detect_plate(vehicle_img)
            if plate_detections:
                plates = plate_recognizer.extract_plate(vehicle_img, plate_detections)
                print(f"Frame {frame_file}, Vehicle {i+1}: Found {len(plates)} plates")
```

### Individual Service Usage

**Vehicle Detection:**
```python
from tattler_vehicle_detection.detector import VehicleDetector

detector = VehicleDetector()
detections = detector.detect_vehicles("parking_lot.jpg")
annotated = detector.visualize_detections("parking_lot.jpg", detections)
```

**Plate Recognition:**
```python
from tattler_plate_recognition.recognizer import PlateRecognizer

recognizer = PlateRecognizer()
detections = recognizer.detect_plate("car.jpg")
annotated = recognizer.annotate_plates("car.jpg", detections)
```

## 🧪 Testing

Each service includes comprehensive test suites:

```bash
# Test vehicle detection
cd services/vehicle_detection
./run_tests.sh

# Test plate recognition
cd services/plate_recognition
./run_tests.sh
```

Add test images to `tests/input/` directories and run the tests to see the services in action.

## 🐳 Docker Deployment

### Build Services
```bash
# Build from repository root
docker build -f services/vehicle_detection/Dockerfile -t tattler-vehicle-detection .
docker build -f services/plate_recognition/Dockerfile -t tattler-plate-recognition .
docker build -f services/ffmpeg_extraction_job/Dockerfile -t tattler-ffmpeg .
```

### Run Services
```bash
# Vehicle detection service
docker run -v $(pwd)/images:/app/images tattler-vehicle-detection

# Plate recognition service
docker run -v $(pwd)/images:/app/images tattler-plate-recognition

# FFmpeg extraction service
docker run -v $(pwd)/videos:/app/videos -v $(pwd)/frames:/app/frames tattler-ffmpeg
```

## 📊 Performance

### Benchmarks (CPU - Intel i7)
- **Vehicle Detection**: ~50ms per image (YOLOv8n)
- **Plate Recognition**: ~100ms per image (YOLO-v9-s)
- **Frame Extraction**: ~30 FPS processing speed

### GPU Acceleration
- **Vehicle Detection**: ~15ms per image (CUDA)
- **Plate Recognition**: ~30ms per image (CUDA)

## 🔧 Configuration

### Model Selection
```python
# Fast inference (recommended for real-time)
detector = VehicleDetector(model_size='n')  # nano

# High accuracy (recommended for analysis)
detector = VehicleDetector(model_size='x')  # extra-large
```

### Confidence Thresholds
```python
# Conservative detection (fewer false positives)
detector = VehicleDetector(confidence_threshold=0.7)

# Sensitive detection (catch more objects)
detector = VehicleDetector(confidence_threshold=0.3)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check service-specific README files
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

## 🗺️ Roadmap

- [ ] **Real-time Processing**: Live video stream support
- [ ] **Web Dashboard**: Browser-based monitoring interface
- [ ] **Database Integration**: Store detection results
- [ ] **Alert System**: Notifications for specific events
- [ ] **Analytics**: Parking lot utilization statistics
- [ ] **Mobile App**: Remote monitoring capabilities

## 🏷️ Version History

- **v0.1.0** - Initial release with core detection services
- **v0.2.0** - Added Docker support and improved packaging
- **v0.3.0** - Enhanced test suites and documentation

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the vehicle detection model
- **YOLO-v9**: open-image-models for license plate detection
- **FFmpeg**: For high-performance video processing
- **OpenCV**: Computer vision utilities

---

**Tattler** - Making parking lot monitoring intelligent and automated 🚁📹🤖
