#!/usr/bin/env python3
"""
Standalone test for license plate recognition service.
Usage: python test_standalone.py [image_path]
"""
import sys
import os
import json
from ocr import LicensePlateRecognizer

def test_license_plate_recognition(image_path: str = None):
    """Test license plate recognition with a single image."""
    
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
    print("LICENSE PLATE RECOGNITION STANDALONE TEST")
    print("=" * 60)
    print(f"Testing image: {image_path}")
    print()
    
    try:
        # Initialize recognizer
        print("Initializing Tesseract OCR recognizer...")
        recognizer = LicensePlateRecognizer()
        print("✅ Recognizer initialized successfully")
        print()
        
        # Run OCR
        print("Running license plate recognition...")
        result = recognizer.extract_text_from_image(image_path)
        
        # Display results
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ OCR completed successfully")
        print()
        print("RESULTS:")
        print("-" * 40)
        print(f"Image size: {result['image_size']['width']}x{result['image_size']['height']}")
        print(f"Raw OCR text: '{result['raw_text']}'")
        print(f"Cleaned text: '{result['cleaned_text']}'")
        print(f"Character count: {result['character_count']}")
        print(f"OCR confidence: {result['confidence']}%")
        print(f"Valid license plate: {result['is_valid_plate']}")
        print()
        
        print("PROCESSING DETAILS:")
        print("-" * 40)
        for step, description in result['processing_steps'].items():
            print(f"  {step}: {description}")
        print()
        
        # Check for GPS data
        print("Checking for GPS data in EXIF...")
        gps_data = recognizer.get_gps_from_exif(image_path)
        if gps_data:
            print(f"✅ GPS coordinates found: {gps_data['lat']:.6f}, {gps_data['lon']:.6f}")
        else:
            print("ℹ️  No GPS data found in image EXIF")
        print()
        
        # Create preprocessing visualization
        print("Creating preprocessing visualization...")
        vis_path = recognizer.visualize_preprocessing(image_path)
        if vis_path:
            print(f"✅ Preprocessing visualization saved to: {vis_path}")
        
        # Save results to JSON
        results_path = image_path.replace('.jpg', '_ocr_results.json').replace('.jpeg', '_ocr_results.json').replace('.png', '_ocr_results.json')
        full_result = result.copy()
        full_result['gps_data'] = gps_data
        
        with open(results_path, 'w') as f:
            json.dump(full_result, f, indent=2)
        print(f"✅ Results saved to: {results_path}")
        
        print()
        print("=" * 60)
        if result['is_valid_plate'] and result['cleaned_text']:
            print(f"🎉 POTENTIAL LICENSE PLATE DETECTED: {result['cleaned_text']}")
        else:
            print("ℹ️  NO VALID LICENSE PLATE DETECTED")
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
    success = test_license_plate_recognition(image_path)
    sys.exit(0 if success else 1)
