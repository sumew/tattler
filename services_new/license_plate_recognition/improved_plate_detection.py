"""
Improved License Plate Region Detection
"""
import cv2
import numpy as np
from typing import List, Tuple
import math

class ImprovedPlateDetector:
    def __init__(self):
        """Initialize the improved plate detector."""
        pass
    
    def find_license_plate_regions_improved(self, vehicle_crop: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find potential license plate regions using multiple advanced techniques.
        
        Args:
            vehicle_crop: Cropped vehicle image
            
        Returns:
            List of (x, y, w, h) tuples for potential license plate regions
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []
        
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY) if len(vehicle_crop.shape) == 3 else vehicle_crop
        h, w = gray.shape
        
        # Combine multiple detection methods
        regions = []
        
        # Method 1: Morphological operations for rectangular regions
        regions.extend(self._detect_with_morphology(gray))
        
        # Method 2: Adaptive thresholding + contour analysis
        regions.extend(self._detect_with_adaptive_threshold(gray))
        
        # Method 3: Sobel edge detection for horizontal edges
        regions.extend(self._detect_with_sobel(gray))
        
        # Method 4: Template matching approach
        regions.extend(self._detect_with_template_matching(gray))
        
        # Filter and merge overlapping regions
        filtered_regions = self._filter_and_merge_regions(regions, w, h)
        
        return filtered_regions
    
    def _detect_with_morphology(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect license plates using morphological operations."""
        regions = []
        
        # Create morphological kernel for license plate shape (rectangular)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        
        # Apply morphological operations
        morph = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._is_valid_plate_region(x, y, w, h, gray.shape[1], gray.shape[0]):
                regions.append((x, y, w, h))
        
        return regions
    
    def _detect_with_adaptive_threshold(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect license plates using adaptive thresholding."""
        regions = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Additional validation for license plate characteristics
            if self._is_valid_plate_region(x, y, w, h, gray.shape[1], gray.shape[0]):
                # Check if the region has text-like characteristics
                roi = gray[y:y+h, x:x+w]
                if self._has_text_characteristics(roi):
                    regions.append((x, y, w, h))
        
        return regions
    
    def _detect_with_sobel(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect license plates using Sobel edge detection."""
        regions = []
        
        # Apply Sobel operator to find horizontal edges (license plates have strong horizontal edges)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
        
        # Focus on horizontal edges (license plates typically have strong horizontal edges)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        horizontal_edges = cv2.morphologyEx(sobel_combined, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Threshold
        _, thresh = cv2.threshold(horizontal_edges, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._is_valid_plate_region(x, y, w, h, gray.shape[1], gray.shape[0]):
                regions.append((x, y, w, h))
        
        return regions
    
    def _detect_with_template_matching(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect license plates using template matching approach."""
        regions = []
        
        # Create a simple license plate template (white rectangle with black border)
        template_sizes = [(60, 20), (80, 25), (100, 30), (120, 35)]
        
        for template_w, template_h in template_sizes:
            if template_w > gray.shape[1] or template_h > gray.shape[0]:
                continue
                
            # Create template
            template = np.ones((template_h, template_w), dtype=np.uint8) * 255
            cv2.rectangle(template, (2, 2), (template_w-3, template_h-3), 0, 2)
            
            # Template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            threshold = 0.3
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                w, h = template_w, template_h
                if self._is_valid_plate_region(x, y, w, h, gray.shape[1], gray.shape[0]):
                    regions.append((x, y, w, h))
        
        return regions
    
    def _is_valid_plate_region(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> bool:
        """
        Validate if a region could be a license plate based on size and aspect ratio.
        """
        # Basic size constraints
        if w < 30 or h < 10 or w > img_w * 0.8 or h > img_h * 0.5:
            return False
        
        # Aspect ratio constraints (license plates are typically 2:1 to 5:1)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 1.5 or aspect_ratio > 6.0:
            return False
        
        # Area constraints
        area = w * h
        img_area = img_w * img_h
        if area < img_area * 0.005 or area > img_area * 0.3:  # 0.5% to 30% of image
            return False
        
        # Position constraints (license plates are usually in lower 2/3 of vehicle)
        if y < img_h * 0.1:  # Not in the very top of the vehicle
            return False
        
        return True
    
    def _has_text_characteristics(self, roi: np.ndarray) -> bool:
        """
        Check if a region has characteristics typical of text/license plates.
        """
        if roi.size == 0:
            return False
        
        # Check for sufficient contrast variation (text regions have high variation)
        std_dev = np.std(roi)
        if std_dev < 20:  # Low contrast regions are unlikely to be license plates
            return False
        
        # Check for horizontal patterns (license plates have horizontal text)
        horizontal_projection = np.sum(roi, axis=1)
        horizontal_variation = np.std(horizontal_projection)
        
        vertical_projection = np.sum(roi, axis=0)
        vertical_variation = np.std(vertical_projection)
        
        # License plates typically have more horizontal variation than vertical
        if horizontal_variation == 0 or vertical_variation / horizontal_variation > 2:
            return False
        
        return True
    
    def _filter_and_merge_regions(self, regions: List[Tuple[int, int, int, int]], 
                                 img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        """
        Filter overlapping regions and merge nearby ones.
        """
        if not regions:
            return []
        
        # Remove duplicates and sort by area (largest first)
        unique_regions = list(set(regions))
        unique_regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        
        # Merge overlapping regions
        merged_regions = []
        
        for region in unique_regions:
            x, y, w, h = region
            
            # Check if this region overlaps significantly with any existing merged region
            overlaps = False
            for i, merged in enumerate(merged_regions):
                mx, my, mw, mh = merged
                
                # Calculate overlap
                overlap_x = max(0, min(x + w, mx + mw) - max(x, mx))
                overlap_y = max(0, min(y + h, my + mh) - max(y, my))
                overlap_area = overlap_x * overlap_y
                
                region_area = w * h
                merged_area = mw * mh
                
                # If overlap is significant, merge the regions
                if overlap_area > 0.3 * min(region_area, merged_area):
                    # Merge by taking the bounding box of both regions
                    new_x = min(x, mx)
                    new_y = min(y, my)
                    new_w = max(x + w, mx + mw) - new_x
                    new_h = max(y + h, my + mh) - new_y
                    
                    merged_regions[i] = (new_x, new_y, new_w, new_h)
                    overlaps = True
                    break
            
            if not overlaps:
                merged_regions.append(region)
        
        # Final validation and sorting
        final_regions = []
        for region in merged_regions:
            x, y, w, h = region
            if self._is_valid_plate_region(x, y, w, h, img_w, img_h):
                final_regions.append(region)
        
        # Sort by confidence (area and position)
        final_regions.sort(key=lambda r: (r[2] * r[3], -r[1]), reverse=True)
        
        # Limit to top 10 regions to avoid too many false positives
        return final_regions[:10]
