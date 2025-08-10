# Vehicle Detection Service

AI-powered vehicle detection using YOLO (You Only Look Once) deep learning model. This service identifies and extracts vehicles from images with high accuracy and speed.

## Features

- **Multi-vehicle Detection**: Detects cars, motorcycles, buses, and trucks
- **High Performance**: Uses optimized YOLOv8 model for fast inference
- **Flexible Input**: Supports both file paths and numpy arrays
- **Vehicle Extraction**: Crops individual vehicle regions from images
- **Visualization**: Creates annotated images with bounding boxes

## Installation

```bash
# Install from the tattler root directory
pip install -e .
pip install -e ./services/vehicle_detection
```

## Quick Start

```python
from tattler_vehicle_detection.detector import VehicleDetector

# Initialize detector
detector = VehicleDetector()

# Detect vehicles in an image
detections = detector.detect_vehicles("path/to/image.jpg")

# Extract individual vehicle images
vehicle_crops = detector.extract_vehicle_images("path/to/image.jpg", detections)

# Create visualization with bounding boxes
annotated_image = detector.visualize_detections("path/to/image.jpg", detections)
```

## API Reference

### VehicleDetector

#### `__init__(model_size='n', confidence_threshold=0.5)`
Initialize the vehicle detector.

**Parameters:**
- `model_size` (str): YOLO model size ('n', 's', 'm', 'l', 'x'). Default: 'n'
- `confidence_threshold` (float): Minimum confidence for detections. Default: 0.5

#### `detect_vehicles(image_input)`
Detect vehicles in an image.

**Parameters:**
- `image_input` (str | np.ndarray): Image file path or numpy array

**Returns:**
- `DetectionResult`: Dictionary containing detection information

**Example Response:**
```python
{
    "image_size": {"width": 1920, "height": 1080},
    "vehicles_detected": 3,
    "detections": [
        {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.89,
            "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
            "area": 40000
        }
    ],
    "has_vehicles": True
}
```

#### `extract_vehicle_images(image_input, detection_result)`
Extract individual vehicle regions from the image.

**Parameters:**
- `image_input` (str | np.ndarray): Image file path or numpy array
- `detection_result` (DetectionResult): Result from `detect_vehicles()`

**Returns:**
- `List[np.ndarray]`: List of cropped vehicle images

#### `visualize_detections(image_input, detection_result)`
Create visualization with bounding boxes.

**Parameters:**
- `image_input` (str | np.ndarray): Image file path or numpy array
- `detection_result` (DetectionResult): Result from `detect_vehicles()`

**Returns:**
- `np.ndarray`: Annotated image with bounding boxes

## Supported Vehicle Types

- **Cars** (class_id: 2)
- **Motorcycles** (class_id: 3)
- **Buses** (class_id: 5)
- **Trucks** (class_id: 7)

## Testing

```bash
# Add test images to tests/input/
cp your_images.jpg tests/input/

# Run tests
python tests/test_extract_vehicles.py
```

## Model Information

- **Architecture**: YOLOv8n (nano version for speed)
- **Training Data**: COCO dataset
- **Input Size**: 640x640 pixels
- **Inference Time**: ~50ms per image (CPU)

## Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy shared utilities
COPY src/tattler/ ./src/tattler/
COPY pyproject.toml ./
RUN pip install -e .

# Copy service
COPY services/vehicle_detection/ ./
RUN pip install -e .

CMD ["python", "-m", "tattler_vehicle_detection.main"]
```

## Performance Tips

1. **Model Size**: Use 'n' for speed, 'x' for accuracy
2. **Batch Processing**: Process multiple images together when possible
3. **Image Size**: Resize large images to reduce processing time
4. **Confidence Threshold**: Adjust based on your accuracy requirements

## Troubleshooting

### Model Loading Issues
- Ensure `yolov8n.pt` model file is present
- Check internet connection for first-time model download

### Memory Issues
- Use smaller model size ('n' instead of 'x')
- Process images in smaller batches
- Resize input images if very large

### Low Detection Accuracy
- Increase confidence threshold
- Use larger model size ('m', 'l', or 'x')
- Ensure good image quality and lighting
