#!/usr/bin/env python3
"""
Standalone test for vehicle detection service.
Usage: python test_standalone.py [image_path]
"""
import sys
import os
import json
from detector import VehicleDetector

def test_vehicle_detection(image_path: str = None):
    """Test vehicle detection with a single image."""
    
    # Use default test image if none provided
    if image_path is None:
        # Look for test images in the detected_frames_local directory
        test_dirs = [
            "../../detected_frames_local",
            "../../../detected_frames_local", 
            "../../tests/sample_images"
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    image_path = os.path.join(test_dir, images[0])
                    print(f"Using test image: {image_path}")
                    break
        
        if image_path is None:
            print("No test image found. Please provide an image path.")
            print("Usage: python test_standalone.py path/to/image.jpg")
            return False
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    print("=" * 60)
    print("VEHICLE DETECTION STANDALONE TEST")
    print("=" * 60)
    print(f"Testing image: {image_path}")
    print()
    
    try:
        # Initialize detector
        print("Initializing YOLO vehicle detector...")
        detector = VehicleDetector(model_size='n', confidence_threshold=0.5)
        print("✅ Detector initialized successfully")
        print()
        
        # Run detection
        print("Running vehicle detection...")
        result = detector.detect_vehicles(image_path)
        
        # Display results
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ Detection completed successfully")
        print()
        print("RESULTS:")
        print("-" * 40)
        print(f"Image size: {result['image_size']['width']}x{result['image_size']['height']}")
        print(f"Vehicles detected: {result['vehicles_detected']}")
        print(f"Has vehicles: {result['has_vehicles']}")
        print()
        
        if result['detections']:
            print("DETECTIONS:")
            for i, detection in enumerate(result['detections'], 1):
                print(f"  {i}. {detection['class_name'].upper()}")
                print(f"     Confidence: {detection['confidence']}")
                print(f"     Bounding box: ({detection['bbox']['x1']}, {detection['bbox']['y1']}) to ({detection['bbox']['x2']}, {detection['bbox']['y2']})")
                print(f"     Area: {detection['area']} pixels")
                print()
        else:
            print("No vehicles detected in this image.")
        
        # Create visualization
        print("Creating visualization...")
        vis_path = detector.visualize_detections(image_path)
        if vis_path:
            print(f"✅ Visualization saved to: {vis_path}")
        
        # Save results to JSON
        results_path = image_path.replace('.jpg', '_detection_results.json').replace('.jpeg', '_detection_results.json').replace('.png', '_detection_results.json')
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Results saved to: {results_path}")
        
        print()
        print("=" * 60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_vehicle_detection(image_path)
    sys.exit(0 if success else 1)
