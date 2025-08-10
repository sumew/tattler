#!/bin/bash

# Run tests for Plate Recognition Service
# This script runs the plate recognition test

set -e  # Exit on any error

echo "🧪 Running Plate Recognition Tests"
echo "=================================="
echo

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "tests/test_plate_recognition.py" ]; then
    echo "❌ Error: Must be run from the plate_recognition service directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected to find: tests/test_plate_recognition.py"
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
    echo "⚠️  No input images found in tests/input/"
    echo "   Please add some images to test with:"
    echo "   - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif"
    echo "   - Place images in: tests/input/"
    echo
    echo "🔄 Running test anyway (will show 'no images found' message)..."
    echo
fi

# Run the test
echo -e "${BLUE}🚀 Starting plate recognition test...${NC}"
echo

python tests/test_plate_recognition.py

echo
echo -e "${GREEN}✅ Test completed!${NC}"
echo
echo "📋 Results:"
echo "   Input images: tests/input/"
echo "   Output files: tests/output/"
echo "   - *_annotated.jpg (images with plate outlines)"
echo "   - *_plate_*.jpg (individual cropped plates)"
echo
