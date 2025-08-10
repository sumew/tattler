#!/usr/bin/env python3
"""
Test for extracting vehicle images from input folder
Processes all images in tests/input/ and saves extracted vehicles to tests/output/
"""

import os
import cv2
from pathlib import Path

# Import from the tattler vehicle detection package
from tattler_vehicle_detection.detector import VehicleDetector

def test_extract_vehicles():
    """
    Process all images in tests/input folder and extract vehicle images
    """
    # Setup paths
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Initialize detector
    print("🤖 Initializing vehicle detector...")
    detector = VehicleDetector()
    
    # Get all image files from input directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"⚠️  No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} image(s) to process")
    
    # Process each image
    for image_file in image_files:
        print(f"\n🔍 Processing: {image_file.name}")
        
        # Read image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"❌ Could not read image: {image_file.name}")
            continue
        
        # Detect vehicles
        detection_result = detector.detect_vehicles(image)
        
        if "error" in detection_result:
            print(f"❌ Detection error: {detection_result['error']}")
            continue
        
        # Extract vehicle images
        vehicle_crops = detector.extract_vehicle_images(image, detection_result)
        
        if not vehicle_crops:
            print(f"⚠️  No vehicles detected in {image_file.name}")
            continue
        
        # Save extracted vehicles
        base_name = image_file.stem
        for i, vehicle_crop in enumerate(vehicle_crops):
            output_filename = f"{base_name}_vehicle_{i+1}.jpg"
            output_path = output_dir / output_filename
            
            success = cv2.imwrite(str(output_path), vehicle_crop)
            if success:
                print(f"💾 Saved: {output_filename}")
            else:
                print(f"❌ Failed to save: {output_filename}")
        
        print(f"✅ Extracted {len(vehicle_crops)} vehicle(s) from {image_file.name}")
    
    print(f"\n🎉 Processing complete! Check {output_dir} for extracted vehicle images.")

if __name__ == "__main__":
    test_extract_vehicles()
