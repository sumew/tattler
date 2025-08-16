# Plate Reader Service

Optical Character Recognition (OCR) service for extracting text from license plate images using fast-plate-ocr, a specialized deep learning model trained specifically for license plates.

## Features

- **Specialized License Plate OCR**: Uses fast-plate-ocr models trained specifically on license plate data
- **High Accuracy**: Deep learning models optimized for license plate recognition
- **Fast Inference**: Extremely fast processing (~0.3ms per image)
- **Global Support**: Works with license plates from different countries
- **No Preprocessing**: Works directly on cropped license plate images
- **Multiple Models**: Choose between speed (XS) and accuracy (S) models

## Installation

```bash
# Install from the tattler root directory
pip install -e .
pip install -e ./services/plate_reader

# Dependencies are automatically installed:
# - fast-plate-ocr (specialized license plate OCR)
```

## Quick Start

```python
from tattler_plate_reader.reader import PlateReader

# Initialize reader (uses fastest model by default)
reader = PlateReader()

# Extract text from license plate image
plate_text = reader.read_plate_text("license_plate.jpg")
print(f"License plate: {plate_text}")

# Get detailed results
detailed = reader.read_plate_text_detailed("license_plate.jpg")
print(f"Text: {detailed['text']}, Confidence: {detailed['confidence']:.2f}")
```

## API Reference

### PlateReader

#### `__init__(model_name="cct-xs-v1-global-model")`
Initialize the plate reader.

**Parameters:**
- `model_name` (str): fast-plate-ocr model to use. Default: "cct-xs-v1-global-model"

**Available Models:**
- `"cct-xs-v1-global-model"` - Fastest model (0.32ms, 3094 plates/sec)
- `"cct-s-v1-global-model"` - More accurate model (0.59ms, 1701 plates/sec)

**Example:**
```python
# Use fastest model (default)
reader = PlateReader()

# Use more accurate model
reader = PlateReader(model_name="cct-s-v1-global-model")
```

#### `read_plate_text(image_input)`
Extract text from a license plate image.

**Parameters:**
- `image_input` (str | np.ndarray): Image file path or numpy array

**Returns:**
- `str | None`: Extracted license plate text, or None if no text found

**Example:**
```python
# From file path
text = reader.read_plate_text("plate.jpg")

# From numpy array
import cv2
image = cv2.imread("plate.jpg")
text = reader.read_plate_text(image)

if text:
    print(f"License plate: {text}")
else:
    print("No text detected")
```

#### `read_plate_text_detailed(image_input)`
Extract text with detailed OCR results.

**Parameters:**
- `image_input` (str | np.ndarray): Image file path or numpy array

**Returns:**
- `Dict[str, any]`: Detailed OCR results

**Example Response:**
```python
{
    "text": "ABC123",
    "confidence": 0.89,
    "raw_text": "A B C 1 2 3",
    "ocr_engine": "easyocr",
    "image_size": {"width": 300, "height": 100},
    "processed_size": {"width": 600, "height": 200},
    "all_detections": [  # EasyOCR only
        {"text": "ABC123", "confidence": 0.89, "bbox": [[x1,y1], [x2,y2], ...]}
    ]
}
```

## Available Models

fast-plate-ocr provides pre-trained models optimized for different use cases:

| Model Name | Size | Speed (ms) | Throughput (plates/sec) | Use Case |
|------------|------|------------|-------------------------|----------|
| `cct-xs-v1-global-model` | XS | 0.32 | 3,094 | **Default** - Fastest processing |
| `cct-s-v1-global-model` | S | 0.59 | 1,701 | Higher accuracy |

### Model Selection Guide

- **Use XS model** (default) for real-time processing and high throughput
- **Use S model** for maximum accuracy when speed is less critical
- Both models work globally with license plates from different countries

## Usage Examples

### Basic Text Extraction
```python
from tattler_plate_reader.reader import PlateReader

reader = PlateReader()

# Process single image
text = reader.read_plate_text("license_plate.jpg")
if text:
    print(f"Detected: {text}")
```

### Using Different Models
```python
# Use fastest model (default)
reader = PlateReader()

# Use more accurate model
reader = PlateReader(model_name="cct-s-v1-global-model")
```

### Batch Processing
```python
import os
from pathlib import Path

reader = PlateReader()
input_dir = Path("plate_images")
results = {}

for image_file in input_dir.glob("*.jpg"):
    text = reader.read_plate_text(str(image_file))
    results[image_file.name] = text
    print(f"{image_file.name}: {text or 'No text detected'}")
```

### Integration with Plate Recognition
```python
from tattler_plate_recognition.recognizer import PlateRecognizer
from tattler_plate_reader.reader import PlateReader

# Step 1: Detect license plates
recognizer = PlateRecognizer()
detections = recognizer.detect_plate("car_image.jpg")

# Step 2: Extract plate regions
plate_crops = recognizer.extract_plate("car_image.jpg", detections)

# Step 3: Read text from each plate
reader = PlateReader()
for i, plate_img in enumerate(plate_crops):
    text = reader.read_plate_text(plate_img)
    print(f"Plate {i+1}: {text or 'Unreadable'}")
```

### Complete Vehicle Processing Pipeline
```python
from tattler_vehicle_detection.detector import VehicleDetector
from tattler_plate_recognition.recognizer import PlateRecognizer
from tattler_plate_reader.reader import PlateReader

# Initialize all services
vehicle_detector = VehicleDetector()
plate_recognizer = PlateRecognizer()
plate_reader = PlateReader()

# Process image
image_path = "parking_lot.jpg"

# Step 1: Detect vehicles
vehicle_detections = vehicle_detector.detect_vehicles(image_path)
vehicle_crops = vehicle_detector.extract_vehicle_images(image_path, vehicle_detections)

# Step 2: For each vehicle, detect and read license plates
for i, vehicle_img in enumerate(vehicle_crops):
    print(f"\nVehicle {i+1}:")
    
    # Detect plates in vehicle
    plate_detections = plate_recognizer.detect_plate(vehicle_img)
    
    if plate_detections:
        # Extract plate regions
        plate_crops = plate_recognizer.extract_plate(vehicle_img, plate_detections)
        
        # Read text from each plate
        for j, plate_img in enumerate(plate_crops):
            text = plate_reader.read_plate_text(plate_img)
            print(f"  Plate {j+1}: {text or 'Unreadable'}")
    else:
        print("  No license plates detected")
```

## Testing

The service includes a comprehensive test suite:

```bash
# Add cropped license plate images to the input directory
cp your_plate_images.jpg tests/input/

# Run the test suite
./run_tests.sh
```

### Test Output
- `plate_reading_results.txt` - Detailed OCR results for all processed images

## Image Requirements

### Best Results
- **Cropped plates**: Images should contain only the license plate
- **Good contrast**: Dark text on light background (or vice versa)
- **Sufficient resolution**: At least 150x50 pixels
- **Clear text**: Minimal blur, good lighting
- **Straight angle**: Minimal perspective distortion

### Preprocessing
The service automatically applies:
- **Grayscale conversion**
- **Image resizing** (if too small)
- **Noise reduction** (Gaussian blur)
- **Contrast enhancement** (adaptive thresholding)
- **Morphological cleaning**

## Performance Considerations

### Speed
- **EasyOCR**: ~1-3 seconds per image (first run slower due to model loading)
- **Tesseract**: ~0.1-0.5 seconds per image
- **Preprocessing**: ~10-50ms per image

### Accuracy Tips
1. **Use high-quality images** with good contrast
2. **Crop tightly** around the license plate
3. **Ensure proper lighting** - avoid shadows and glare
4. **Try different OCR engines** if one doesn't work well
5. **Consider image orientation** - plates should be right-side up

## Docker Deployment

```dockerfile
FROM python:3.10-slim

# Install system dependencies for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared utilities
COPY src/tattler/ ./src/tattler/
COPY pyproject.toml ./
RUN pip install -e .

# Copy service
COPY services/plate_reader/ ./
RUN pip install -e .

CMD ["python", "-m", "tattler_plate_reader.main"]
```

## Troubleshooting

### OCR Engine Issues

#### EasyOCR Not Working
```
ImportError: EasyOCR not available
```
**Solution**: `pip install easyocr`

#### Tesseract Not Found
```
TesseractNotFoundError: tesseract is not installed
```
**Solution**: Install Tesseract binary:
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt install tesseract-ocr`
- Windows: Download from GitHub releases

### Poor OCR Results

#### No Text Detected
- Check image quality and contrast
- Ensure the image contains a license plate
- Try the other OCR engine
- Verify image is not corrupted

#### Incorrect Text
- Improve image quality (resolution, lighting)
- Crop more tightly around the plate
- Try different OCR engine
- Check for perspective distortion

#### Low Confidence Scores
- Enhance image contrast
- Increase image resolution
- Remove noise and blur
- Ensure proper cropping

### Memory Issues
```
CUDA out of memory (EasyOCR)
```
**Solution**: Use CPU mode (default) or reduce batch size

## Configuration

### Custom OCR Settings

#### EasyOCR Configuration
```python
# Custom EasyOCR settings
reader = PlateReader(
    ocr_engine="easyocr",
    languages=["en"]  # Add more languages as needed
)
```

#### Tesseract Configuration
```python
# The service uses optimized Tesseract config:
# --psm 8: Single word mode
# Character whitelist: A-Z, 0-9 only
```

### Environment Variables
```bash
# Set Tesseract path (if not in system PATH)
export TESSERACT_CMD=/usr/local/bin/tesseract

# EasyOCR model directory
export EASYOCR_MODULE_PATH=/path/to/models
```

## Integration Examples

### With Web API
```python
from flask import Flask, request, jsonify
from tattler_plate_reader.reader import PlateReader

app = Flask(__name__)
reader = PlateReader()

@app.route('/read_plate', methods=['POST'])
def read_plate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    result = reader.read_plate_text_detailed(image)
    return jsonify(result)
```

### With Database Storage
```python
import sqlite3
from datetime import datetime

def process_and_store_plates(image_paths):
    reader = PlateReader()
    conn = sqlite3.connect('plates.db')
    
    for image_path in image_paths:
        result = reader.read_plate_text_detailed(image_path)
        
        if result['text']:
            conn.execute('''
                INSERT INTO plates (text, confidence, timestamp, image_path)
                VALUES (?, ?, ?, ?)
            ''', (result['text'], result['confidence'], datetime.now(), image_path))
    
    conn.commit()
    conn.close()
```

## Best Practices

1. **Preprocess images** before OCR for better results
2. **Use appropriate OCR engine** based on your needs
3. **Validate results** using confidence scores
4. **Handle errors gracefully** when OCR fails
5. **Cache OCR models** to avoid repeated loading
6. **Monitor performance** and adjust settings as needed
