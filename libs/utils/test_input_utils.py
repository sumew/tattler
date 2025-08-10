#!/usr/bin/env python3
"""
Simple test for input_utils to verify the refactoring works
"""

import numpy as np
import cv2
import tempfile
import os
from tattler.utils.input_utils import load_image_from_input

def test_input_utils():
    """Test the load_image_from_input function"""
    print("🧪 Testing input_utils.load_image_from_input()")
    
    # Test 1: Valid numpy array
    print("\n1. Testing with valid numpy array...")
    test_array = np.ones((100, 100, 3), dtype=np.uint8) * 128
    image, path, error = load_image_from_input(test_array)
    
    if error is None and image is not None and path is None:
        print("   ✅ Numpy array input: PASS")
    else:
        print(f"   ❌ Numpy array input: FAIL - {error}")
    
    # Test 2: Empty numpy array
    print("\n2. Testing with empty numpy array...")
    empty_array = np.array([])
    image, path, error = load_image_from_input(empty_array)
    
    if error is not None and "Empty image array" in error:
        print("   ✅ Empty array handling: PASS")
    else:
        print(f"   ❌ Empty array handling: FAIL - Expected error, got {error}")
    
    # Test 3: Invalid input type
    print("\n3. Testing with invalid input type...")
    image, path, error = load_image_from_input(123)
    
    if error is not None and "Invalid input type" in error:
        print("   ✅ Invalid type handling: PASS")
    else:
        print(f"   ❌ Invalid type handling: FAIL - Expected error, got {error}")
    
    # Test 4: Non-existent file path
    print("\n4. Testing with non-existent file...")
    image, path, error = load_image_from_input("nonexistent_file.jpg")
    
    if error is not None and "not found" in error:
        print("   ✅ Non-existent file handling: PASS")
    else:
        print(f"   ❌ Non-existent file handling: FAIL - Expected error, got {error}")
    
    # Test 5: Valid file path (create temporary image)
    print("\n5. Testing with valid file path...")
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        # Create a simple test image
        test_img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        cv2.imwrite(tmp_file.name, test_img)
        
        # Test loading
        image, path, error = load_image_from_input(tmp_file.name)
        
        if error is None and image is not None and path == tmp_file.name:
            print("   ✅ Valid file path: PASS")
        else:
            print(f"   ❌ Valid file path: FAIL - {error}")
        
        # Clean up
        os.unlink(tmp_file.name)
    
    print("\n🎉 Input utils test complete!")

if __name__ == "__main__":
    test_input_utils()
