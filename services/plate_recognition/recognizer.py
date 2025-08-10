from open_image_models import LicensePlateDetector
from open_image_models.detection.core.hub import PlateDetectorModel
from open_image_models.detection.core.base import DetectionResult

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, TypedDict
import os

# Add libs to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs"))
from utils.input_utils import load_image_from_input

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

    def draw_detections(self, image: np.ndarray, detections: List[DetectionResult], 
                       color: Tuple[int, int, int] = (0, 255, 0), 
                       thickness: int = 2,
                       font_scale: float = 0.6) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image for detected plates.

        Args:
            image: Input image (BGR format)
            detections: List of DetectionResult instances
            color: BGR color for bounding boxes (default: green)
            thickness: Line thickness for bounding boxes
            font_scale: Font scale for labels

        Returns:
            Image with drawn bounding boxes and labels
        """
        # Create a copy to avoid modifying the original
        annotated_image = image.copy()
        
        for i, detection in enumerate(detections):
            # Get bounding box coordinates
            bbox = detection.bounding_box
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            confidence = detection.confidence
            label = f"Plate {i+1}: {confidence:.2f}"
            
            # Calculate label position (above the bounding box)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
            
            # Draw label background (optional, for better visibility)
            cv2.rectangle(annotated_image, 
                         (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0] + 5, label_y + 5), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1 + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Print detection info
            print(f"  📍 Plate {i+1}: confidence={confidence:.3f}, "
                  f"bbox=({x1}, {y1}, {x2}, {y2}), "
                  f"size={x2-x1}x{y2-y1}")
        
        return annotated_image

    def save_annotated_image(self, image: np.ndarray, detections: List[DetectionResult], 
                           output_path: str, **draw_kwargs) -> str:
        """
        Draw detections on image and save to file.

        Args:
            image: Input image (BGR format)
            detections: List of DetectionResult instances
            output_path: Path to save the annotated image
            **draw_kwargs: Additional arguments for draw_detections

        Returns:
            Path to saved image
        """
        # Draw detections
        annotated_image = self.draw_detections(image, detections, **draw_kwargs)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        success = cv2.imwrite(output_path, annotated_image)
        
        if success:
            print(f"💾 Annotated image saved to: {output_path}")
            return output_path
        else:
            print(f"❌ Failed to save image to: {output_path}")
            return ""

    def extract_plate_regions(self, image: np.ndarray, detections: List[DetectionResult], 
                            output_dir: str = "extracted_plates") -> List[str]:
        """
        Extract individual license plate regions and save them as separate images.

        Args:
            image: Input image (BGR format)
            detections: List of DetectionResult instances
            output_dir: Directory to save extracted plate images

        Returns:
            List of paths to extracted plate images
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        
        for i, detection in enumerate(detections):
            # Get bounding box coordinates
            bbox = detection.bounding_box
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            
            # Extract plate region
            plate_region = image[y1:y2, x1:x2]
            
            # Generate filename
            confidence = detection.confidence
            filename = f"plate_{i+1}_conf_{confidence:.3f}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # Save extracted plate
            success = cv2.imwrite(output_path, plate_region)
            
            if success:
                extracted_files.append(output_path)
                print(f"📁 Extracted plate {i+1} to: {output_path}")
            else:
                print(f"❌ Failed to extract plate {i+1}")
        
        return extracted_files

    def process_image_with_visualization(self, image_input: Union[str, np.ndarray], 
                                       output_dir: str = "results") -> Dict[str, any]:
        """
        Complete processing pipeline: detect plates, draw annotations, save results.

        Args:
            image_input: Either a file path (str) or image array (np.ndarray)
            output_dir: Directory to save results

        Returns:
            Dictionary with processing results
        """
        print(f"🔍 Processing image for license plate detection...")
        
        # Load image if path provided
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                return {"error": f"Could not load image: {image_input}"}
            base_name = Path(image_input).stem
        else:
            image = image_input
            base_name = "image"
        
        # Detect plates
        detections = self.detect_plate(image)
        
        if not detections:
            print("⚠️  No license plates detected")
            return {
                "detections": [],
                "plates_found": 0,
                "annotated_image_path": "",
                "extracted_plates": []
            }
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save annotated image
        annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
        self.save_annotated_image(image, detections, annotated_path)
        
        # Extract individual plates
        plates_dir = os.path.join(output_dir, "extracted_plates")
        extracted_plates = self.extract_plate_regions(image, detections, plates_dir)
        
        # Prepare results
        results = {
            "detections": detections,
            "plates_found": len(detections),
            "annotated_image_path": annotated_path,
            "extracted_plates": extracted_plates,
            "image_size": {"width": image.shape[1], "height": image.shape[0]}
        }
        
        print(f"✅ Processing complete! Found {len(detections)} plates")
        print(f"📁 Results saved to: {output_dir}")
        
        return results

    def get_detection_info(self, detections: List[DetectionResult]) -> List[Dict[str, any]]:
        """
        Extract information from DetectionResult instances in a readable format.

        Args:
            detections: List of DetectionResult instances

        Returns:
            List of dictionaries with detection information
        """
        detection_info = []
        
        for i, detection in enumerate(detections):
            bbox = detection.bounding_box
            info = {
                "plate_id": i + 1,
                "label": detection.label,
                "confidence": detection.confidence,
                "bounding_box": {
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                    "x2": bbox.x2,
                    "y2": bbox.y2
                },
                "width": bbox.x2 - bbox.x1,
                "height": bbox.y2 - bbox.y1,
                "area": (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
            }
            detection_info.append(info)
        
        return detection_info
