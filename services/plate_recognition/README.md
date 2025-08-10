# Plate Recognition Service

AI-powered license plate detection service using advanced YOLO models. This service identifies license plates in images and provides tools for annotation and extraction.

## Features

- **High-Accuracy Detection**: Uses YOLO-v9 model optimized for license plates
- **Flexible Input**: Supports both file paths and numpy arrays
- **Plate Annotation**: Draw outlines around detected plates
- **Plate Extraction**: Crop individual license plate regions
- **Simple API**: Clean, focused interface with three main methods

## Installation

```bash
# Install from the tattler root directory
pip install -e .
pip install -e ./services/plate_recognition
```

## Quick Start

```python
from tattler_plate_recognition.recognizer import PlateRecognizer

# Initialize recognizer
recognizer = PlateRecognizer()

# Detect license plates
detections = recognizer.detect_plate("path/to/image.jpg")

# Create annotated image with plate outlines
annotated_image = recognizer.annotate_plates("path/to/image.jpg", detections)

# Extract individual plate images
plate_crops = recognizer.extract_plate("path/to/image.jpg", detections)
```

## API Reference

### PlateRecognizer

#### `__init__(confidence_threshold=0.5)`
Initialize the plate recognizer.

**Parameters:**
- `confidence_threshold` (float): Minimum confidence for detections. Default: 0.5

#### `detect_plate(image_input)`
Detect license plates in an image.

**Parameters:**
- `image_input` (str | np.ndarray): Image file path or numpy array

**Returns:**
- `List[DetectionResult]`: List of detection results from the model

**Example Usage:**
```python
detections = recognizer.detect_plate("car_image.jpg")
print(f"Found {len(detections)} license plates")

for detection in detections:
    bbox = detection.bounding_box
    confidence = detection.confidence
    print(f"Plate at ({bbox.x1}, {bbox.y1}) with confidence {confidence:.2f}")
```

#### `annotate_plates(image_input, detections, color=(0,255,0), thickness=2)`
Draw outlines around detected license plates.

**Parameters:**
- `image_input` (str | np.ndarray): Image file path or numpy array
- `detections` (List[DetectionResult]): Results from `detect_plate()`
- `color` (Tuple[int,int,int]): BGR color for outlines. Default: green (0,255,0)
- `thickness` (int): Line thickness for outlines. Default: 2

**Returns:**
- `np.ndarray | None`: Annotated image with plate outlines, or None if error

**Example Usage:**
```python
# Green outlines (default)
annotated = recognizer.annotate_plates(image, detections)

# Red outlines with thicker lines
annotated = recognizer.annotate_plates(image, detections, color=(0,0,255), thickness=3)
```

#### `extract_plate(image_input, detections)`
Extract individual license plate regions from the image.

**Parameters:**
- `image_input` (str | np.ndarray): Image file path or numpy array
- `detections` (List[DetectionResult]): Results from `detect_plate()`

**Returns:**
- `List[np.ndarray]`: List of cropped license plate images

**Example Usage:**
```python
plate_crops = recognizer.extract_plate(image, detections)

for i, plate_img in enumerate(plate_crops):
    cv2.imwrite(f"plate_{i+1}.jpg", plate_img)
    print(f"Saved plate {i+1} with shape: {plate_img.shape}")
```

## Complete Workflow Example

```python
import cv2
from tattler_plate_recognition.recognizer import PlateRecognizer

# Initialize
recognizer = PlateRecognizer()

# Load image
image_path = "parking_lot.jpg"
image = cv2.imread(image_path)

# Step 1: Detect plates
print("🔍 Detecting license plates...")
detections = recognizer.detect_plate(image_path)

if detections:
    print(f"✅ Found {len(detections)} license plate(s)")
    
    # Step 2: Create annotated image
    annotated = recognizer.annotate_plates(image_path, detections)
    cv2.imwrite("annotated_result.jpg", annotated)
    
    # Step 3: Extract individual plates
    plates = recognizer.extract_plate(image_path, detections)
    for i, plate in enumerate(plates):
        cv2.imwrite(f"extracted_plate_{i+1}.jpg", plate)
    
    print("🎉 Processing complete!")
else:
    print("⚠️  No license plates detected")
```

## Testing

The service includes a comprehensive test suite:

```bash
# Add test images to the input directory
cp your_car_images.jpg tests/input/

# Run the test suite
./run_tests.sh
```

### Test Output
- `*_annotated.jpg` - Original images with green rectangles around plates
- `*_plate_*.jpg` - Individual cropped license plate images

## Model Information

- **Architecture**: YOLO-v9-s (small version optimized for license plates)
- **Model Name**: `yolo-v9-s-608-license-plate-end2end`
- **Input Size**: 608x608 pixels
- **Training**: Specialized dataset for license plate detection
- **Provider**: open-image-models library

## Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy shared utilities
COPY src/tattler/ ./src/tattler/
COPY pyproject.toml ./
RUN pip install -e .

# Copy service
COPY services/plate_recognition/ ./
RUN pip install -e .

CMD ["python", "-m", "tattler_plate_recognition.main"]
```

## Performance Considerations

### Speed Optimization
- Model loads once during initialization
- Subsequent detections are fast (~100-200ms per image)
- Batch processing multiple images is more efficient

### Accuracy Tips
- **Good Lighting**: Ensure plates are well-lit and visible
- **Image Quality**: Higher resolution images generally work better
- **Angle**: Front-facing or rear-facing plates work best
- **Distance**: Plates should be reasonably large in the image

### Memory Usage
- Model requires ~50MB RAM when loaded
- Processing memory scales with image size
- Consider resizing very large images for better performance

## Troubleshooting

### Model Loading Issues
```
❌ Error loading detector: ...
```
- Check internet connection (model downloads on first use)
- Verify `open-image-models` package is installed
- Clear model cache: `rm -rf ~/.cache/open-image-models/`

### No Detections Found
```
🔍 Found 0 license plate(s)
```
- Check image quality and lighting
- Ensure license plates are visible and not obscured
- Try lowering confidence threshold: `PlateRecognizer(confidence_threshold=0.3)`
- Verify image format is supported

### Import Errors
```
ModuleNotFoundError: No module named 'tattler_plate_recognition'
```
- Ensure packages are installed: `pip install -e . && pip install -e ./services/plate_recognition`
- Check you're in the correct directory
- Verify virtual environment is activated

## Integration Examples

### With Vehicle Detection
```python
from tattler_vehicle_detection.detector import VehicleDetector
from tattler_plate_recognition.recognizer import PlateRecognizer

# Detect vehicles first
vehicle_detector = VehicleDetector()
vehicle_detections = vehicle_detector.detect_vehicles(image)
vehicle_crops = vehicle_detector.extract_vehicle_images(image, vehicle_detections)

# Then detect plates in each vehicle
plate_recognizer = PlateRecognizer()
for i, vehicle_img in enumerate(vehicle_crops):
    plate_detections = plate_recognizer.detect_plate(vehicle_img)
    if plate_detections:
        plates = plate_recognizer.extract_plate(vehicle_img, plate_detections)
        print(f"Vehicle {i+1}: Found {len(plates)} license plate(s)")
```

### Batch Processing
```python
import os
from pathlib import Path

recognizer = PlateRecognizer()
input_dir = Path("input_images")
output_dir = Path("results")

for image_file in input_dir.glob("*.jpg"):
    detections = recognizer.detect_plate(str(image_file))
    
    if detections:
        # Save annotated image
        annotated = recognizer.annotate_plates(str(image_file), detections)
        cv2.imwrite(str(output_dir / f"{image_file.stem}_annotated.jpg"), annotated)
        
        # Save extracted plates
        plates = recognizer.extract_plate(str(image_file), detections)
        for j, plate in enumerate(plates):
            cv2.imwrite(str(output_dir / f"{image_file.stem}_plate_{j+1}.jpg"), plate)
```
