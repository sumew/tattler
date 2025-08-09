#!/usr/bin/env python3
"""
Test the integrated Vehicle Detection + License Plate Recognition pipeline.
Usage: python test_integrated.py [image_path]
"""
import sys
import os
import json
from integrated_pipeline import IntegratedVehicleLPRPipeline

def test_integrated_pipeline(image_path: str = None):
    """Test the integrated pipeline with a single image."""
    
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
            print("Usage: python test_integrated.py path/to/image.jpg")
            return False
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    print("=" * 80)
    print("INTEGRATED VEHICLE DETECTION + LICENSE PLATE RECOGNITION TEST")
    print("=" * 80)
    print(f"Testing image: {image_path}")
    print()
    
    try:
        # Initialize integrated pipeline
        print("🚀 Initializing integrated pipeline...")
        pipeline = IntegratedVehicleLPRPipeline(
            vehicle_confidence=0.5,  # Minimum confidence for vehicle detection
            ocr_confidence=20        # Minimum confidence for OCR (lowered for testing)
        )
        print()
        
        # Run integrated processing
        print("🔄 Running integrated processing...")
        results = pipeline.process_image(image_path)
        
        # Check for errors
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return False
        
        print("\n" + "=" * 80)
        print("📊 INTEGRATED RESULTS SUMMARY")
        print("=" * 80)
        
        # Display summary
        summary = results['processing_summary']
        print(f"Image size: {results['image_size']['width']}x{results['image_size']['height']}")
        print(f"Vehicles detected: {results['vehicles_detected']}")
        print(f"License plates found: {results['license_plates_found']}")
        print(f"Vehicles with plates: {summary['vehicles_with_plates']}")
        print(f"Vehicle confidence threshold: {summary['vehicle_confidence_threshold']}")
        print(f"OCR confidence threshold: {summary['ocr_confidence_threshold']}")
        print()
        
        # Display vehicle details
        if results['vehicle_detections']:
            print("🚗 VEHICLE DETECTIONS:")
            print("-" * 50)
            for i, vehicle in enumerate(results['vehicle_detections'], 1):
                bbox = vehicle['bbox']
                print(f"  {i}. {vehicle['class_name'].upper()}")
                print(f"     Confidence: {vehicle['confidence']}")
                print(f"     Location: ({bbox['x1']}, {bbox['y1']}) to ({bbox['x2']}, {bbox['y2']})")
                print(f"     Size: {bbox['x2']-bbox['x1']}x{bbox['y2']-bbox['y1']} pixels")
                print()
        
        # Display license plate details
        if results['license_plates']:
            print("🔍 LICENSE PLATE DETECTIONS:")
            print("-" * 50)
            for i, plate in enumerate(results['license_plates'], 1):
                print(f"  {i}. LICENSE PLATE: '{plate['text']}'")
                print(f"     Confidence: {plate['confidence']}%")
                print(f"     Detection method: {plate['method']}")
                print(f"     Raw OCR text: '{plate['raw_text']}'")
                if 'plate_region' in plate:
                    region = plate['plate_region']
                    print(f"     Plate region: {region['x']},{region['y']} ({region['w']}x{region['h']})")
                print()
        else:
            print("❌ No license plates detected")
            print()
            print("💡 TROUBLESHOOTING TIPS:")
            print("   - License plates might be too small in drone footage")
            print("   - Try lowering OCR confidence threshold")
            print("   - Check if vehicles are close enough to camera")
            print("   - Ensure license plates are clearly visible")
            print()
        
        # Display per-vehicle analysis
        print("📋 PER-VEHICLE ANALYSIS:")
        print("-" * 50)
        for vehicle_info in results['vehicle_crops_analysis']:
            print(f"Vehicle {vehicle_info['vehicle_id']}: {vehicle_info['vehicle_class']} "
                  f"(confidence: {vehicle_info['vehicle_confidence']})")
            
            if vehicle_info['license_plates']:
                for plate in vehicle_info['license_plates']:
                    print(f"  ✅ Found plate: '{plate['text']}' ({plate['confidence']}% confidence)")
            else:
                print(f"  ❌ No license plates detected")
            print()
        
        # Create visualizations
        print("🎨 Creating visualizations...")
        
        # 1. Basic integrated visualization
        vis_path = pipeline.create_visualization(image_path, results)
        if vis_path:
            print(f"✅ Basic visualization saved to: {vis_path}")
        
        # 2. Detailed analysis with license plate regions
        detailed_vis_path = pipeline.create_detailed_visualization(image_path, results)
        if detailed_vis_path:
            print(f"✅ Detailed analysis saved to: {detailed_vis_path}")
        
        # 3. Individual vehicle crop analysis
        vehicle_crops_paths = pipeline.create_vehicle_crops_visualization(image_path, results)
        if vehicle_crops_paths:
            print(f"✅ Individual vehicle analysis saved ({len(vehicle_crops_paths)} files)")
            print(f"   Directory: {os.path.dirname(vehicle_crops_paths[0])}")
        
        # Save detailed results to JSON
        results_path = image_path.replace('.jpg', '_integrated_results.json').replace('.jpeg', '_integrated_results.json').replace('.png', '_integrated_results.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        json_safe_results = convert_numpy_types(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        print(f"✅ Detailed results saved to: {results_path}")
        
        print("\n" + "=" * 80)
        if results['license_plates_found'] > 0:
            print("🎉 SUCCESS: License plates detected using integrated pipeline!")
            for plate in results['license_plates']:
                print(f"   📋 {plate['text']} (confidence: {plate['confidence']}%)")
        else:
            print("ℹ️  No license plates detected in this image")
            print("   This is normal for drone footage where plates may be too small/distant")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_integrated_pipeline(image_path)
    sys.exit(0 if success else 1)
