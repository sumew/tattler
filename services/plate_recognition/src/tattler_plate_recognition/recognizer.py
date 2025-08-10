from open_image_models import LicensePlateDetector
from open_image_models.detection.core.hub import PlateDetectorModel
from open_image_models.detection.core.base import DetectionResult

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, TypedDict
import os

# Import from tattler shared utilities
from tattler.utils.input_utils import load_image_from_input

class PlateRecognizer:
    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """
        Initialize the plate recognizer.

        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold: float = confidence_threshold
        self._load_model()

    def _load_model(self) -> None:
        """Load the model."""
        try:
            print(f"🔄 Initializing detector...")
            self.detector = LicensePlateDetector(detection_model="yolo-v9-s-608-license-plate-end2end")
            print(f"✅ Detector initialized successfully")
        except Exception as e:
            print(f"❌ Error loading detector: {e}")
            import traceback
            traceback.print_exc()
            raise

    def detect_plate(self, image_input: Union[str, np.ndarray]) -> List[DetectionResult]:
        """
        Detect plates in an image.

        Args:
            image_input: Either a file path (str) or image array (np.ndarray)

        Returns:
            List of DetectionResult instances
        """
        try:
            # Handle different input types using utility
            image, image_path, error = load_image_from_input(image_input)
            if error:
                print(f"❌ {error}")
                return []

            # Convert BGR to RGB for the detector (if needed)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume input is BGR (OpenCV format), convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Run detection
            detections = self.detector.predict(image_rgb)
            
            print(f"🔍 Found {len(detections)} license plate(s)")
            return detections

        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def annotate_plates(self, image_input: Union[str, np.ndarray], 
                       detections: List[DetectionResult],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> Optional[np.ndarray]:
        """
        Draw outlines around detected license plates on the image.

        Args:
            image_input: Either a file path (str) or image array (np.ndarray)
            detections: List of DetectionResult instances from detect_plate()
            color: BGR color for bounding boxes (default: green)
            thickness: Line thickness for bounding boxes

        Returns:
            Image array with plate outlines drawn, or None if error
        """
        try:
            # Handle different input types using utility
            image, image_path, error = load_image_from_input(image_input)
            if error:
                print(f"❌ {error}")
                return None

            # Create a copy to avoid modifying the original
            annotated_image = image.copy()
            
            # Draw bounding boxes for each detection
            for detection in detections:
                # Get bounding box coordinates
                bbox = detection.bounding_box
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                
                # Draw rectangle outline
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            return annotated_image

        except Exception as e:
            print(f"❌ Error annotating image: {str(e)}")
            return None

    def extract_plate(self, image_input: Union[str, np.ndarray], 
                     detections: List[DetectionResult]) -> List[np.ndarray]:
        """
        Extract individual license plate regions from the image.

        Args:
            image_input: Either a file path (str) or image array (np.ndarray)
            detections: List of DetectionResult instances from detect_plate()

        Returns:
            List of numpy arrays, each containing a cropped license plate image
        """
        try:
            # Handle different input types using utility
            image, image_path, error = load_image_from_input(image_input)
            if error:
                print(f"❌ {error}")
                return []

            extracted_plates = []
            
            # Extract each detected plate region
            for i, detection in enumerate(detections):
                # Get bounding box coordinates
                bbox = detection.bounding_box
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                
                # Extract plate region
                plate_region = image[y1:y2, x1:x2]
                
                # Add to results if the crop is valid
                if plate_region.size > 0:
                    extracted_plates.append(plate_region)
                else:
                    print(f"⚠️  Skipping invalid plate region {i+1}")
            
            print(f"📁 Extracted {len(extracted_plates)} license plate region(s)")
            return extracted_plates

        except Exception as e:
            print(f"❌ Error extracting plates: {str(e)}")
            return []
