"""
Input handling utilities
Common functions for processing different input types (file paths vs numpy arrays)
"""

import cv2
import numpy as np
import os
from typing import Union, Tuple, Optional
from pathlib import Path


def load_image_from_input(image_input: Union[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    Handle different input types and load image consistently.
    
    Args:
        image_input: Either a file path (str) or image array (np.ndarray)
        
    Returns:
        Tuple of (image_array, image_path, error_message)
        - image_array: Loaded image as numpy array, None if failed
        - image_path: Path to image file if input was string, None if input was array
        - error_message: Error description if loading failed, None if successful
    """
    try:
        if isinstance(image_input, str):
            # Input is a file path
            image_path = image_input
            
            # Check if file exists
            if not os.path.exists(image_path):
                return None, image_path, f"Image file not found: {image_path}"
            
            # Load image from file
            image = cv2.imread(image_path)
            if image is None:
                return None, image_path, f"Could not read image: {image_path}"
            
            return image, image_path, None
        
        elif isinstance(image_input, np.ndarray):
            # Input is already an image array
            image = image_input
            
            # Validate image array
            if image.size == 0:
                return None, None, "Empty image array provided"
            
            # Basic shape validation
            if len(image.shape) not in [2, 3]:
                return None, None, f"Invalid image shape: {image.shape}. Expected 2D or 3D array"
            
            return image, None, None
        
        else:
            return None, None, f"Invalid input type: {type(image_input)}. Expected str or np.ndarray"
            
    except Exception as e:
        return None, None, f"Error processing input: {str(e)}"
