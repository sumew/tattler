"""
Vehicle Detection Module using YOLO
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import os

# YOLO vehicle classes (COCO dataset class IDs)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle', 
    5: 'bus',
    7: 'truck'
}

class VehicleDetector:
    def __init__(self, model_size: str = 'n', confidence_threshold: float = 0.5):
        """
        Initialize the vehicle detector.
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = f'yolov8{model_size}.pt'
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
    
    def detect_vehicles(self, image_path: str) -> Dict:
        """
        Detect vehicles in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing detection results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not read image: {image_path}"}
            
            # Run YOLO detection
            results = self.model(image, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a vehicle class with good confidence
                        if class_id in VEHICLE_CLASSES and confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            detection = {
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
            
            return {
                "image_path": image_path,
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "vehicles_detected": len(detections),
                "detections": detections,
                "has_vehicles": len(detections) > 0
            }
            
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
    
    def detect_vehicles_from_array(self, image: np.ndarray) -> Dict:
        """
        Detect vehicles in an image array.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Run YOLO detection
            results = self.model(image, verbose=False)
            
            # Process results (same logic as above)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id in VEHICLE_CLASSES and confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            detection = {
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
            
            return {
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "vehicles_detected": len(detections),
                "detections": detections,
                "has_vehicles": len(detections) > 0
            }
            
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
    
    def visualize_detections(self, image_path: str, output_path: str = None) -> str:
        """
        Create a visualization of the detections.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            
        Returns:
            Path to the visualization image
        """
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_detected{ext}"
        
        # Get detections
        result = self.detect_vehicles(image_path)
        if "error" in result:
            print(f"Error: {result['error']}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        
        # Draw bounding boxes
        for detection in result['detections']:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(image, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, 
                       (bbox['x1'], bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to: {output_path}")
        return output_path
