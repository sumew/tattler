#!/usr/bin/env python3
"""
Test for plate reader service
Processes all images in tests/input/ and extracts license plate text
"""

import os
import cv2
from pathlib import Path

# Import from the tattler plate reader package
from tattler_plate_reader.reader import PlateReader

def test_plate_reader():
    """
    Process all images in tests/input folder and extract license plate text
    """
    # Setup paths
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Initialize reader
    print("🔄 Initializing plate reader...")
    try:
        reader = PlateReader()  # Uses fast-plate-ocr
    except Exception as e:
        print(f"❌ fast-plate-ocr initialization failed: {e}")
        return
    
    # Get all image files from input directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"⚠️  No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} image(s) to process")
    
    # Create results file
    results_file = output_dir / "plate_reading_results.txt"
    
    with open(results_file, 'w') as f:
        f.write("License Plate Reading Results\n")
        f.write("=" * 40 + "\n\n")
        
        # Process each image
        for image_file in image_files:
            print(f"\n🔍 Processing: {image_file.name}")
            f.write(f"Image: {image_file.name}\n")
            
            # Read image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"❌ Could not read image: {image_file.name}")
                f.write("Error: Could not read image\n\n")
                continue
            
            # Extract text using simple method
            plate_text = reader.read_plate_text(image)
            
            if plate_text:
                print(f"✅ Extracted text: '{plate_text}'")
                f.write(f"Extracted Text: {plate_text}\n")
            else:
                print(f"⚠️  No text detected in {image_file.name}")
                f.write("Extracted Text: [No text detected]\n")
            
            # Get detailed results
            detailed_results = reader.read_plate_text_detailed(image)
            
            f.write(f"OCR Engine: {detailed_results.get('ocr_engine', 'unknown')}\n")
            f.write(f"Confidence: {detailed_results.get('confidence', 0):.2f}\n")
            f.write(f"Raw Text: '{detailed_results.get('raw_text', '')}'\n")
            f.write(f"Image Size: {detailed_results.get('image_size', {})}\n")
            f.write("-" * 30 + "\n\n")
            
            print(f"📊 Confidence: {detailed_results.get('confidence', 0):.2f}")
    
    print(f"\n🎉 Processing complete!")
    print(f"📄 Results saved to: {results_file}")
    print(f"📁 Check {output_dir} for detailed results.")

if __name__ == "__main__":
    test_plate_reader()
