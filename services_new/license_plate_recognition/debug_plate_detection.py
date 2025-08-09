#!/usr/bin/env python3
"""
Debug script to visualize improved license plate detection.
Usage: python debug_plate_detection.py [image_path]
"""
import sys
import os
import cv2
import numpy as np
sys.path.append('../vehicle_detection')
from detector import VehicleDetector
from improved_plate_detection import ImprovedPlateDetector

def debug_plate_detection(image_path: str):
    """Debug the license plate detection on a single image."""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    print("=" * 80)
    print("LICENSE PLATE DETECTION DEBUG")
    print("=" * 80)
    print(f"Testing image: {image_path}")
    print()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return False
    
    # Initialize detectors
    vehicle_detector = VehicleDetector(confidence_threshold=0.5)
    plate_detector = ImprovedPlateDetector()
    
    # Detect vehicles
    vehicle_results = vehicle_detector.detect_vehicles_from_array(image)
    print(f"Found {vehicle_results['vehicles_detected']} vehicles")
    
    if vehicle_results['vehicles_detected'] == 0:
        print("No vehicles detected - cannot test license plate detection")
        return False
    
    # Process each vehicle
    for i, detection in enumerate(vehicle_results['detections']):
        print(f"\n--- VEHICLE {i+1} ---")
        print(f"Class: {detection['class_name']}")
        print(f"Confidence: {detection['confidence']}")
        
        # Extract vehicle crop
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        vehicle_crop = image[y1:y2, x1:x2]
        
        if vehicle_crop.size == 0:
            print("Empty vehicle crop - skipping")
            continue
        
        print(f"Vehicle crop size: {vehicle_crop.shape[1]}x{vehicle_crop.shape[0]}")
        
        # Find license plate regions with improved detection
        plate_regions = plate_detector.find_license_plate_regions_improved(vehicle_crop)
        print(f"Found {len(plate_regions)} potential license plate regions")
        
        # Create visualization
        vis_crop = vehicle_crop.copy()
        
        # Draw all detected regions
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for j, (px, py, pw, ph) in enumerate(plate_regions):
            color = colors[j % len(colors)]
            cv2.rectangle(vis_crop, (px, py), (px + pw, py + ph), color, 2)
            cv2.putText(vis_crop, f"R{j+1}", (px, py - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            print(f"  Region {j+1}: ({px}, {py}) {pw}x{ph} pixels")
            print(f"    Aspect ratio: {pw/ph:.2f}")
            print(f"    Area: {pw*ph} pixels ({(pw*ph)/(vehicle_crop.shape[0]*vehicle_crop.shape[1])*100:.1f}% of vehicle)")
        
        # Scale up the visualization for better viewing
        scale_factor = max(2, 400 / max(vis_crop.shape[:2]))
        new_size = (int(vis_crop.shape[1] * scale_factor), int(vis_crop.shape[0] * scale_factor))
        vis_crop_large = cv2.resize(vis_crop, new_size, interpolation=cv2.INTER_CUBIC)
        
        # Save individual vehicle debug image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_path = f"{base_name}_vehicle_{i+1}_debug.jpg"
        cv2.imwrite(debug_path, vis_crop_large)
        print(f"  Debug visualization saved: {debug_path}")
        
        # Also save the original crop for comparison
        orig_path = f"{base_name}_vehicle_{i+1}_original.jpg"
        cv2.imwrite(orig_path, vehicle_crop)
        print(f"  Original crop saved: {orig_path}")
    
    # Create overall visualization
    overall_vis = image.copy()
    
    # Draw vehicle bounding boxes
    for i, detection in enumerate(vehicle_results['detections']):
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        cv2.rectangle(overall_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overall_vis, f"V{i+1}: {detection['class_name']}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save overall visualization
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    overall_path = f"{base_name}_debug_overall.jpg"
    cv2.imwrite(overall_path, overall_vis)
    print(f"\nOverall debug visualization saved: {overall_path}")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_plate_detection.py path/to/image.jpg")
        # Use a default image if available
        test_dirs = ["../../detected_frames_local", "../../../detected_frames_local"]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
                if images:
                    image_path = os.path.join(test_dir, images[0])
                    print(f"Using default image: {image_path}")
                    break
        else:
            sys.exit(1)
    else:
        image_path = sys.argv[1]
    
    success = debug_plate_detection(image_path)
    sys.exit(0 if success else 1)
