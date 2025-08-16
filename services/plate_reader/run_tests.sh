#!/bin/bash

# Run tests for Plate Reader Service
# This script runs the plate reader OCR test

set -e  # Exit on any error

echo "🧪 Running Plate Reader Tests"
echo "============================="
echo

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "tests/test_plate_reader.py" ]; then
    echo "❌ Error: Must be run from the plate_reader service directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected to find: tests/test_plate_reader.py"
    exit 1
fi

# Check if input directory exists
if [ ! -d "tests/input" ]; then
    echo "📁 Creating tests/input directory..."
    mkdir -p tests/input
fi

# Check if output directory exists
if [ ! -d "tests/output" ]; then
    echo "📁 Creating tests/output directory..."
    mkdir -p tests/output
fi

# Check for input images
input_count=$(find tests/input -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.bmp" -o -name "*.tiff" -o -name "*.tif" 2>/dev/null | wc -l)

if [ "$input_count" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No input images found in tests/input/${NC}"
    echo "   Please add license plate images to test with:"
    echo "   - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif"
    echo "   - Place cropped license plate images in: tests/input/"
    echo "   - Note: This service works best with cropped plate images"
    echo
    echo "🔄 Running test anyway (will show 'no images found' message)..."
    echo
fi

# Check OCR dependencies
echo -e "${BLUE}🔍 Checking OCR dependencies...${NC}"

# Check for fast-plate-ocr
python -c "import fast_plate_ocr; print('✅ fast-plate-ocr available')" 2>/dev/null || echo "❌ fast-plate-ocr not available (pip install fast-plate-ocr)"

echo

# Run the test
echo -e "${BLUE}🚀 Starting plate reader test...${NC}"
echo

python tests/test_plate_reader.py

echo
echo -e "${GREEN}✅ Test completed!${NC}"
echo
echo "📋 Results:"
echo "   Input images: tests/input/"
echo "   Output files: tests/output/"
echo "   - plate_reading_results.txt (detailed OCR results)"
echo
echo "💡 Tips:"
echo "   - Use cropped license plate images for best results"
echo "   - Ensure good image quality and contrast"
echo "   - Try different models: cct-xs-v1-global-model (fast) or cct-s-v1-global-model (accurate)"
echo
