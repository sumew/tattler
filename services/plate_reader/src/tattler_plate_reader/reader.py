"""
License Plate OCR Reader Module
Extracts text from license plate images using fast-plate-ocr
"""

import cv2
import numpy as np
import re
from typing import Optional, Union, Dict, List

# Import from tattler shared utilities
from tattler.utils.input_utils import load_image_from_input

# Import fast-plate-ocr
try:
    from fast_plate_ocr import LicensePlateRecognizer
    FAST_PLATE_OCR_AVAILABLE = True
except ImportError as e:
    print(f"❌ fast-plate-ocr not available: {e}")
    print("   Install with: pip install fast-plate-ocr")
    FAST_PLATE_OCR_AVAILABLE = False

class PlateReader:
    def __init__(self, model_name: str = "cct-xs-v1-global-model") -> None:
        """
        Initialize the plate reader using fast-plate-ocr.

        Args:
            model_name: Model name for fast-plate-ocr. 
                       Options: "cct-xs-v1-global-model" (fastest), "cct-s-v1-global-model" (more accurate)
                       Default: "cct-xs-v1-global-model"
        """
        self.model_name = model_name
        self.reader = None
        
        self._initialize_ocr()

    def _initialize_ocr(self) -> None:
        """Initialize the fast-plate-ocr engine."""
        if not FAST_PLATE_OCR_AVAILABLE:
            raise ImportError("fast-plate-ocr not available. Install with: pip install fast-plate-ocr")
        
        try:
            print(f"🔄 Initializing fast-plate-ocr with model: {self.model_name}")
            self.reader = LicensePlateRecognizer(hub_ocr_model=self.model_name)
            print("✅ fast-plate-ocr initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing fast-plate-ocr: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common OCR artifacts and padding characters
        cleaned = re.sub(r'[^\w\s]', '', cleaned)  # Remove special characters
        cleaned = re.sub(r'_+', '', cleaned)  # Remove underscore padding (common in fast-plate-ocr)
        cleaned = cleaned.upper()  # Convert to uppercase
        
        # Remove spaces between characters (common in license plates)
        if len(cleaned.replace(' ', '')) <= 10:  # Typical license plate length
            cleaned = cleaned.replace(' ', '')
        
        return cleaned

    def read_plate_text(self, image_input: Union[str, np.ndarray]) -> Optional[str]:
        """
        Extract text from a license plate image using fast-plate-ocr.

        Args:
            image_input: Either a file path (str) or image array (np.ndarray)

        Returns:
            Extracted license plate text as string, or None if no text found
        """
        try:
            # Handle different input types using utility
            image, image_path, error = load_image_from_input(image_input)
            if error:
                print(f"❌ {error}")
                return None

            # Ensure image is in BGR format (fast-plate-ocr expects this)
            if len(image.shape) == 2:
                # Convert grayscale to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # Convert RGBA to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            # Run fast-plate-ocr
            results = self.reader.run(image)
            
            if results and len(results) > 0:
                raw_text = results[0]  # Take the first result
                print(f"🔍 fast-plate-ocr detected: '{raw_text}'")
            else:
                print("⚠️  fast-plate-ocr found no text")
                return None
            
            # Clean and normalize the text
            cleaned_text = self._clean_text(raw_text)
            
            if cleaned_text:
                print(f"✅ Final result: '{cleaned_text}'")
                return cleaned_text
            else:
                print("⚠️  No valid text after cleaning")
                return None

        except Exception as e:
            print(f"❌ Error reading plate text: {str(e)}")
            return None

    def read_plate_text_detailed(self, image_input: Union[str, np.ndarray]) -> Dict[str, any]:
        """
        Extract text from a license plate image with detailed results.

        Args:
            image_input: Either a file path (str) or image array (np.ndarray)

        Returns:
            Dictionary with detailed OCR results
        """
        try:
            # Handle different input types using utility
            image, image_path, error = load_image_from_input(image_input)
            if error:
                return {"error": error, "text": None, "confidence": 0.0}

            # Ensure image is in BGR format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            result = {
                "text": None,
                "confidence": 0.0,
                "raw_text": "",
                "ocr_engine": "fast-plate-ocr",
                "model_name": self.model_name,
                "image_size": {"width": image.shape[1], "height": image.shape[0]}
            }
            
            # Run fast-plate-ocr
            results = self.reader.run(image)
            
            if results and len(results) > 0:
                result["raw_text"] = results[0]
                result["confidence"] = 0.95  # fast-plate-ocr doesn't provide confidence, assume high
                result["all_detections"] = [{"text": text} for text in results]
            
            # Clean the text
            if result["raw_text"]:
                result["text"] = self._clean_text(result["raw_text"])
            
            return result

        except Exception as e:
            return {"error": f"Error reading plate text: {str(e)}", "text": None, "confidence": 0.0}
