"""
Vehicle Detection Module using YOLO
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List, Dict, Tuple, Optional, Union, TypedDict
import os

# Add libs to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs"))
from utils.input_utils import load_image_from_input

# Type definitions for better LSP support
class BoundingBox(TypedDict):
    x1: int
    y1: int
    x2: int
    y2: int

class VehicleDetection(TypedDict):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    area: int

class ImageSize(TypedDict):
    width: int
    height: int

class DetectionResult(TypedDict, total=False):
    # Required fields
    vehicles_detected: int
    detections: List[VehicleDetection]
    has_vehicles: bool
    image_size: ImageSize
    # Optional fields
    image_path: str  # Only present when input is a file path
    error: str  # Only present when there's an error

# YOLO vehicle classes (COCO dataset class IDs)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle', 
    5: 'bus',
    7: 'truck'
}

class VehicleDetector:
    def __init__(self, model_size: str = 'n', confidence_threshold: float = 0.5) -> None:
        """
        Initialize the vehicle detector.

        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold: float = confidence_threshold
        self.model_path: str = f'yolov8{model_size}.pt'
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            print(f"Loading YOLO model: {self.model_path}")

            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            print(f"Model file exists, size: {os.path.getsize(self.model_path)} bytes")

            # Set environment variables to force offline mode
            os.environ['YOLO_CONFIG_DIR'] = '/tmp'
            os.environ['ULTRALYTICS_OFFLINE'] = '1'  # Force offline mode

            # Load the model with explicit local path and offline settings
            print("Initializing YOLO model in offline mode...")
            from ultralytics.utils import SETTINGS
            SETTINGS['sync'] = False  # Disable telemetry/sync

            self.model = YOLO(self.model_path, verbose=False)
            print("✅ YOLO model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def detect_vehicles(self, image_input: Union[str, np.ndarray]) -> DetectionResult:
        """
        Detect vehicles in an image.

        Args:
            image_input: Either a file path (str) or image array (np.ndarray)

        Returns:
            Dictionary containing detection results
        """
        try:
            # Handle different input types using utility
            image, image_path, error = load_image_from_input(image_input)
            if error:
                return {"error": error}
            
            include_path = image_path is not None

            # Run YOLO detection
            results: List[Results] = self.model(image, verbose=False)

            # Process results
            detections: List[VehicleDetection] = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id: int = int(box.cls[0])
                        confidence: float = float(box.conf[0])

                        # Check if it's a vehicle class with good confidence
                        if class_id in VEHICLE_CLASSES and confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            detection: VehicleDetection = {
                                'class_id': class_id,
                                'class_name': VEHICLE_CLASSES[class_id],
                                'confidence': round(confidence, 3),
                                'bbox': {
                                    'x1': int(x1), 'y1': int(y1),
                                    'x2': int(x2), 'y2': int(y2)
                                },
                                'area': int((x2 - x1) * (y2 - y1))
                            }
                            detections.append(detection)

            # Build result dictionary
            result_dict: DetectionResult = {
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "vehicles_detected": len(detections),
                "detections": detections,
                "has_vehicles": len(detections) > 0
            }
            
            # Add image path only if input was a file path
            if include_path and image_path:
                result_dict["image_path"] = image_path

            return result_dict

        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}

    def visualize_detections(self, image_input: Union[str, np.ndarray], 
                           detection_result: DetectionResult) -> Optional[np.ndarray]:
        """
        Create visualization with bounding boxes for detected vehicles.
        
        Args:
            image_input: Either a file path (str) or image array (np.ndarray)
            detection_result: DetectionResult from detect_vehicles method
            
        Returns:
            Image array with bounding boxes drawn, or None if error
        """
        try:
            # Handle different input types using utility
            image, image_path, error = load_image_from_input(image_input)
            if error:
                print(f"Error: {error}")
                return None
            
            # Check for errors in detection result
            if "error" in detection_result:
                print(f"Error: {detection_result['error']}")
                return None

            # Create a copy to avoid modifying the original image
            annotated_image = image.copy()

            # Draw bounding boxes
            for detection in detection_result['detections']:
                bbox = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']

                # Draw rectangle
                cv2.rectangle(annotated_image, 
                             (bbox['x1'], bbox['y1']), 
                             (bbox['x2'], bbox['y2']), 
                             (0, 255, 0), 2)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(annotated_image, label, 
                           (bbox['x1'], bbox['y1'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return annotated_image
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None

    def extract_vehicle_images(self, image_input: Union[str, np.ndarray], 
                             detection_result: DetectionResult) -> List[np.ndarray]:
        """
        Extract individual vehicle images from detections.

        Args:
            image_input: Either a file path (str) or image array (np.ndarray)
            detection_result: DetectionResult from detect_vehicles method

        Returns:
            List of numpy arrays containing cropped vehicle images
        """
        try:
            # Handle different input types using utility
            image, image_path, error = load_image_from_input(image_input)
            if error:
                print(f"Error: {error}")
                return []
            
            # Check for errors in detection result
            if "error" in detection_result:
                print(f"Error: {detection_result['error']}")
                return []

            if not detection_result['has_vehicles']:
                return []

            extracted_crops = []
            for i, detection in enumerate(detection_result['detections']):
                bbox = detection['bbox']
                
                # Extract vehicle region
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                vehicle_crop = image[y1:y2, x1:x2]
                
                # Add to results
                extracted_crops.append(vehicle_crop)

            return extracted_crops

        except Exception as e:
            print(f"Error extracting vehicle images: {str(e)}")
            return []
