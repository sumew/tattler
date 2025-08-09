"""
License Plate Recognition Module using Tesseract OCR
"""
import cv2
import numpy as np
import pytesseract
import piexif
import re
import os
from typing import Dict, Optional, List, Tuple
from PIL import Image

class LicensePlateRecognizer:
    def __init__(self, tesseract_config: str = None):
        """
        Initialize the license plate recognizer.
        
        Args:
            tesseract_config: Custom Tesseract configuration
        """
        self.tesseract_config = tesseract_config or r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self._verify_tesseract()
    
    def _verify_tesseract(self):
        """Verify that Tesseract is available."""
        try:
            pytesseract.get_tesseract_version()
            print("✅ Tesseract OCR is available")
        except Exception as e:
            print(f"❌ Tesseract OCR not found: {e}")
            print("Please install Tesseract OCR:")
            print("  macOS: brew install tesseract")
            print("  Ubuntu: sudo apt-get install tesseract-ocr")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return thresh
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing OCR results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not read image: {image_path}"}
            
            return self.extract_text_from_array(image, image_path)
            
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
    
    def extract_text_from_array(self, image: np.ndarray, image_path: str = "unknown") -> Dict:
        """
        Extract text from an image array using OCR.
        
        Args:
            image: Image as numpy array
            image_path: Path for reference (optional)
            
        Returns:
            Dictionary containing OCR results
        """
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Run OCR
            raw_text = pytesseract.image_to_string(processed, config=self.tesseract_config)
            
            # Clean up the extracted text
            cleaned_text = raw_text.strip().upper()
            cleaned_text = re.sub(r'[^A-Z0-9]', '', cleaned_text)
            
            # Get confidence scores
            data = pytesseract.image_to_data(processed, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Validate if it looks like a license plate
            is_valid_plate = self._validate_license_plate(cleaned_text)
            
            return {
                "image_path": image_path,
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "raw_text": raw_text.strip(),
                "cleaned_text": cleaned_text,
                "is_valid_plate": is_valid_plate,
                "confidence": round(avg_confidence, 2),
                "character_count": len(cleaned_text),
                "processing_steps": {
                    "preprocessing": "Applied contrast enhancement, Gaussian blur, and adaptive thresholding",
                    "ocr_config": self.tesseract_config,
                    "text_cleaning": "Removed non-alphanumeric characters, converted to uppercase"
                }
            }
            
        except Exception as e:
            return {"error": f"Error during OCR processing: {str(e)}"}
    
    def _validate_license_plate(self, text: str) -> bool:
        """
        Validate if the extracted text looks like a license plate.
        
        Args:
            text: Cleaned text from OCR
            
        Returns:
            True if it looks like a valid license plate
        """
        if not text:
            return False
        
        # Basic validation - license plates are typically 3-8 characters
        if len(text) < 3 or len(text) > 8:
            return False
        
        # Should contain both letters and numbers (most license plates do)
        has_letters = bool(re.search(r'[A-Z]', text))
        has_numbers = bool(re.search(r'[0-9]', text))
        
        # At least one letter or number
        return has_letters or has_numbers
    
    def get_gps_from_exif(self, image_path: str) -> Optional[Dict]:
        """
        Extract GPS data from image EXIF.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with GPS coordinates or None
        """
        try:
            exif_dict = piexif.load(image_path)
            
            if "GPS" in exif_dict and exif_dict["GPS"]:
                gps_info = exif_dict["GPS"]
                
                # Check if GPS coordinates are present
                if piexif.GPSIFD.GPSLatitude in gps_info and piexif.GPSIFD.GPSLongitude in gps_info:
                    # Convert GPS coordinates from EXIF format to decimal degrees
                    lat = gps_info[piexif.GPSIFD.GPSLatitude]
                    lat_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef, b'N')
                    lon = gps_info[piexif.GPSIFD.GPSLongitude]
                    lon_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef, b'E')
                    
                    def dms_to_decimal(dms, ref):
                        degrees = float(dms[0][0]) / float(dms[0][1])
                        minutes = float(dms[1][0]) / float(dms[1][1])
                        seconds = float(dms[2][0]) / float(dms[2][1])
                        
                        decimal = degrees + minutes/60 + seconds/3600
                        
                        if ref in [b'S', b'W']:
                            decimal = -decimal
                        
                        return decimal
                    
                    latitude = dms_to_decimal(lat, lat_ref)
                    longitude = dms_to_decimal(lon, lon_ref)
                    
                    return {"lat": latitude, "lon": longitude}
            
            return None
            
        except Exception as e:
            print(f"Error reading EXIF data from {image_path}: {e}")
            return None
    
    def visualize_preprocessing(self, image_path: str, output_path: str = None) -> str:
        """
        Create a visualization showing preprocessing steps.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            
        Returns:
            Path to the visualization image
        """
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_preprocessing{ext}"
        
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Apply preprocessing steps
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
        contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        blurred = cv2.GaussianBlur(contrast, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Create side-by-side comparison
        h, w = original.shape[:2]
        
        # Convert grayscale images to BGR for concatenation
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        contrast_bgr = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # Resize images to fit in a 2x2 grid
        target_size = (w//2, h//2)
        original_small = cv2.resize(original, target_size)
        gray_small = cv2.resize(gray_bgr, target_size)
        contrast_small = cv2.resize(contrast_bgr, target_size)
        thresh_small = cv2.resize(thresh_bgr, target_size)
        
        # Add labels
        def add_label(img, text):
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return img
        
        original_small = add_label(original_small, "Original")
        gray_small = add_label(gray_small, "Grayscale")
        contrast_small = add_label(contrast_small, "Enhanced Contrast")
        thresh_small = add_label(thresh_small, "Thresholded")
        
        # Combine into 2x2 grid
        top_row = np.hstack([original_small, gray_small])
        bottom_row = np.hstack([contrast_small, thresh_small])
        combined = np.vstack([top_row, bottom_row])
        
        # Save visualization
        cv2.imwrite(output_path, combined)
        print(f"Preprocessing visualization saved to: {output_path}")
        return output_path
