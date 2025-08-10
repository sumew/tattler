#!/usr/bin/env python3
"""
Test for plate recognition service
Processes all images in tests/input/ and creates annotated and cropped outputs
"""

import os
import cv2
from pathlib import Path

# Import from the tattler plate recognition package
from tattler_plate_recognition.recognizer import PlateRecognizer

def test_plate_recognition():
    """
    Process all images in tests/input folder and create annotated and cropped outputs
    """
    # Setup paths
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Initialize recognizer
    print("🔄 Initializing plate recognizer...")
    recognizer = PlateRecognizer()
    
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
        
        # Detect plates
        detections = recognizer.detect_plate(image)
        
        if not detections:
            print(f"⚠️  No license plates detected in {image_file.name}")
            continue
        
        base_name = image_file.stem
        
        # Create annotated image
        annotated_image = recognizer.annotate_plates(image, detections)
        if annotated_image is not None:
            annotated_path = output_dir / f"{base_name}_annotated.jpg"
            success = cv2.imwrite(str(annotated_path), annotated_image)
            if success:
                print(f"💾 Saved annotated: {annotated_path.name}")
            else:
                print(f"❌ Failed to save annotated image")
        
        # Extract individual plates
        plate_crops = recognizer.extract_plate(image, detections)
        for i, plate_crop in enumerate(plate_crops):
            crop_path = output_dir / f"{base_name}_plate_{i+1}.jpg"
            success = cv2.imwrite(str(crop_path), plate_crop)
            if success:
                print(f"📁 Saved plate crop: {crop_path.name}")
            else:
                print(f"❌ Failed to save plate crop {i+1}")
        
        print(f"✅ Processed {image_file.name}: {len(detections)} plate(s) detected")
    
    print(f"\n🎉 Processing complete! Check {output_dir} for results.")

if __name__ == "__main__":
    test_plate_recognition()
