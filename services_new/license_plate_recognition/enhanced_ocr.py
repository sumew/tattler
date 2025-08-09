"""
Enhanced OCR specifically for license plate regions
"""
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple, Optional

class EnhancedLicensePlateOCR:
    def __init__(self):
        """Initialize the enhanced OCR system."""
        # Multiple OCR configurations to try
        self.ocr_configs = [
            # Standard license plate config
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # Single text line
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # Single word
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # Raw line (no assumptions)
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # Sparse text
            r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]
    
    def enhance_license_plate_region(self, plate_region: np.ndarray, aggressive: bool = True) -> List[np.ndarray]:
        """
        Apply multiple enhancement techniques to a license plate region.
        
        Args:
            plate_region: The cropped license plate region
            aggressive: Whether to apply aggressive enhancement
            
        Returns:
            List of enhanced versions to try OCR on
        """
        if plate_region is None or plate_region.size == 0:
            return []
        
        enhanced_versions = []
        
        # Convert to grayscale if needed
        if len(plate_region.shape) == 3:
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_region.copy()
        
        # 1. Massive upscaling (critical for small license plates)
        scale_factors = [4, 6, 8] if aggressive else [3, 4]
        for scale in scale_factors:
            height, width = gray.shape
            new_size = (width * scale, height * scale)
            upscaled = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)
            enhanced_versions.append(self._apply_enhancement_pipeline(upscaled, f"upscaled_{scale}x"))
        
        # 2. Different enhancement pipelines
        enhanced_versions.extend([
            self._enhance_contrast_adaptive(gray),
            self._enhance_with_morphology(gray),
            self._enhance_with_unsharp_mask(gray),
            self._enhance_with_bilateral_filter(gray),
            self._enhance_with_denoising(gray),
        ])
        
        if aggressive:
            enhanced_versions.extend([
                self._enhance_with_histogram_equalization(gray),
                self._enhance_with_gamma_correction(gray, 0.5),
                self._enhance_with_gamma_correction(gray, 1.5),
                self._enhance_with_edge_enhancement(gray),
            ])
        
        return enhanced_versions
    
    def _apply_enhancement_pipeline(self, image: np.ndarray, method_name: str) -> np.ndarray:
        """Apply a comprehensive enhancement pipeline."""
        # Step 1: Noise reduction
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Step 2: Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(denoised)
        
        # Step 3: Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        
        # Step 4: Final cleanup
        final = cv2.GaussianBlur(sharpened, (3, 3), 0)
        
        return final
    
    def _enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Enhance using adaptive histogram equalization."""
        # Upscale first
        upscaled = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(upscaled)
        
        # Sharpen
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _enhance_with_morphology(self, image: np.ndarray) -> np.ndarray:
        """Enhance using morphological operations."""
        upscaled = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
        
        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(upscaled, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(morph, alpha=1.5, beta=0)
        
        return enhanced
    
    def _enhance_with_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """Enhance using unsharp masking."""
        upscaled = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
        
        # Create unsharp mask
        gaussian = cv2.GaussianBlur(upscaled, (9, 9), 10.0)
        unsharp_mask = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
        
        return unsharp_mask
    
    def _enhance_with_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Enhance using bilateral filtering."""
        upscaled = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter multiple times
        filtered = upscaled
        for _ in range(2):
            filtered = cv2.bilateralFilter(filtered, 9, 80, 80)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(filtered, alpha=1.3, beta=10)
        
        return enhanced
    
    def _enhance_with_denoising(self, image: np.ndarray) -> np.ndarray:
        """Enhance using advanced denoising."""
        upscaled = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
        
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(upscaled, None, 10, 7, 21)
        
        # Sharpen after denoising
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def _enhance_with_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Enhance using histogram equalization."""
        upscaled = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(upscaled)
        
        return equalized
    
    def _enhance_with_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Enhance using gamma correction."""
        upscaled = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
        
        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(upscaled, table)
        
        return gamma_corrected
    
    def _enhance_with_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance using edge enhancement."""
        upscaled = cv2.resize(image, (image.shape[1] * 5, image.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
        
        # Detect edges
        edges = cv2.Canny(upscaled, 50, 150)
        
        # Enhance edges
        enhanced = cv2.addWeighted(upscaled, 0.8, edges, 0.2, 0)
        
        return enhanced
    
    def extract_text_from_plate_region(self, plate_region: np.ndarray, debug_save_path: str = None) -> Dict:
        """
        Extract text from a license plate region using multiple enhancement techniques.
        
        Args:
            plate_region: The cropped license plate region
            debug_save_path: Optional path to save debug images
            
        Returns:
            Dictionary with OCR results
        """
        if plate_region is None or plate_region.size == 0:
            return {"error": "Empty plate region"}
        
        print(f"    Trying enhanced OCR on {plate_region.shape[1]}x{plate_region.shape[0]} region...")
        
        # Get multiple enhanced versions
        enhanced_versions = self.enhance_license_plate_region(plate_region, aggressive=True)
        
        best_result = None
        best_confidence = 0
        all_attempts = []
        
        # Try OCR on each enhanced version with each configuration
        for i, enhanced in enumerate(enhanced_versions):
            for j, config in enumerate(self.ocr_configs):
                try:
                    # Run OCR
                    raw_text = pytesseract.image_to_string(enhanced, config=config)
                    cleaned_text = raw_text.strip().upper()
                    cleaned_text = ''.join(c for c in cleaned_text if c.isalnum())
                    
                    # Get confidence
                    try:
                        data = pytesseract.image_to_data(enhanced, config=config, output_type=pytesseract.Output.DICT)
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    except:
                        avg_confidence = 0
                    
                    attempt = {
                        'enhancement_method': f'method_{i}',
                        'ocr_config': j,
                        'raw_text': raw_text.strip(),
                        'cleaned_text': cleaned_text,
                        'confidence': avg_confidence,
                        'char_count': len(cleaned_text)
                    }
                    all_attempts.append(attempt)
                    
                    # Check if this is the best result so far
                    if (len(cleaned_text) >= 3 and  # At least 3 characters
                        avg_confidence > best_confidence and
                        avg_confidence > 10):  # Minimum confidence threshold
                        
                        best_result = attempt
                        best_confidence = avg_confidence
                        
                        # Save the best enhanced version for debugging
                        if debug_save_path:
                            debug_path = debug_save_path.replace('.jpg', f'_best_enhanced.jpg')
                            cv2.imwrite(debug_path, enhanced)
                    
                    # Early exit if we found a very good result
                    if avg_confidence > 70 and len(cleaned_text) >= 4:
                        print(f"    Found high-confidence result: '{cleaned_text}' ({avg_confidence:.1f}%)")
                        break
                        
                except Exception as e:
                    continue
            
            # Early exit if we found a very good result
            if best_confidence > 70:
                break
        
        # Save debug images if requested
        if debug_save_path and enhanced_versions:
            for i, enhanced in enumerate(enhanced_versions[:3]):  # Save first 3 enhanced versions
                debug_path = debug_save_path.replace('.jpg', f'_enhanced_{i}.jpg')
                cv2.imwrite(debug_path, enhanced)
        
        # Return the best result or a summary
        if best_result:
            result = {
                'text': best_result['cleaned_text'],
                'confidence': best_result['confidence'],
                'raw_text': best_result['raw_text'],
                'enhancement_method': best_result['enhancement_method'],
                'ocr_config': best_result['ocr_config'],
                'total_attempts': len(all_attempts),
                'is_valid_plate': self._validate_license_plate_text(best_result['cleaned_text'])
            }
            print(f"    Best result: '{result['text']}' (confidence: {result['confidence']:.1f}%)")
            return result
        else:
            # Return summary of all attempts
            if all_attempts:
                best_attempt = max(all_attempts, key=lambda x: x['confidence'])
                result = {
                    'text': best_attempt['cleaned_text'],
                    'confidence': best_attempt['confidence'],
                    'raw_text': best_attempt['raw_text'],
                    'total_attempts': len(all_attempts),
                    'is_valid_plate': False,
                    'all_attempts_summary': [
                        f"'{a['cleaned_text']}' ({a['confidence']:.1f}%)" 
                        for a in sorted(all_attempts, key=lambda x: x['confidence'], reverse=True)[:5]
                    ]
                }
                print(f"    No high-confidence result. Best attempt: '{result['text']}' ({result['confidence']:.1f}%)")
                return result
            else:
                return {
                    'text': '',
                    'confidence': 0,
                    'raw_text': '',
                    'total_attempts': 0,
                    'is_valid_plate': False,
                    'error': 'All OCR attempts failed'
                }
    
    def _validate_license_plate_text(self, text: str) -> bool:
        """Validate if the extracted text looks like a license plate."""
        if not text or len(text) < 3 or len(text) > 8:
            return False
        
        # Should contain both letters and numbers (most license plates do)
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        # At least one letter or number
        return has_letters or has_numbers
