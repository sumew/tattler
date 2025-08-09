"""
Integrated Vehicle Detection + License Plate Recognition Pipeline
"""
import cv2
import numpy as np
import os
import sys
import json
from typing import Dict, List, Tuple, Optional

# Add the vehicle detection path to import the detector
sys.path.append('../vehicle_detection')
from detector import VehicleDetector
from ocr import LicensePlateRecognizer
from improved_plate_detection import ImprovedPlateDetector
from enhanced_ocr import EnhancedLicensePlateOCR

class IntegratedVehicleLPRPipeline:
    def __init__(self, vehicle_confidence: float = 0.5, ocr_confidence: float = 15):
        """
        Initialize the integrated pipeline.
        
        Args:
            vehicle_confidence: Minimum confidence for vehicle detection
            ocr_confidence: Minimum confidence for OCR results (lowered for enhanced OCR)
        """
        self.vehicle_confidence = vehicle_confidence
        self.ocr_confidence = ocr_confidence
        
        print("Initializing integrated pipeline...")
        self.vehicle_detector = VehicleDetector(confidence_threshold=vehicle_confidence)
        self.lpr = LicensePlateRecognizer()
        self.improved_plate_detector = ImprovedPlateDetector()
        self.enhanced_ocr = EnhancedLicensePlateOCR()
        print("✅ Pipeline initialized successfully")
    
    def enhance_vehicle_crop_for_ocr(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """
        Enhance a vehicle crop specifically for license plate OCR.
        
        Args:
            vehicle_crop: Cropped vehicle image
            
        Returns:
            Enhanced image optimized for OCR
        """
        # Resize to make license plates larger (important for OCR)
        height, width = vehicle_crop.shape[:2]
        scale_factor = max(2.0, 400 / max(width, height))  # Ensure minimum 400px on longest side
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized = cv2.resize(vehicle_crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Sharpen the image to make text clearer
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        return filtered
    
    def find_license_plate_regions(self, vehicle_crop: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find potential license plate regions within a vehicle crop using improved detection.
        
        Args:
            vehicle_crop: Cropped vehicle image
            
        Returns:
            List of (x, y, w, h) tuples for potential license plate regions
        """
        return self.improved_plate_detector.find_license_plate_regions_improved(vehicle_crop)
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process an image through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing all results
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        try:
            # Load the original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                return {"error": f"Could not read image: {image_path}"}
            
            print(f"Processing image: {image_path}")
            print(f"Image size: {original_image.shape[1]}x{original_image.shape[0]}")
            
            # Step 1: Detect vehicles
            print("\n🚗 Step 1: Detecting vehicles...")
            vehicle_results = self.vehicle_detector.detect_vehicles_from_array(original_image)
            
            if "error" in vehicle_results:
                return vehicle_results
            
            print(f"Found {vehicle_results['vehicles_detected']} vehicles")
            
            # Step 2: Process each vehicle for license plates
            print("\n🔍 Step 2: Processing vehicles for license plates...")
            
            all_license_plates = []
            vehicle_crops_info = []
            
            for i, detection in enumerate(vehicle_results['detections']):
                print(f"\n  Processing vehicle {i+1}/{len(vehicle_results['detections'])}")
                print(f"  Vehicle: {detection['class_name']} (confidence: {detection['confidence']})")
                
                # Extract vehicle crop
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                vehicle_crop = original_image[y1:y2, x1:x2]
                
                if vehicle_crop.size == 0:
                    continue
                
                # Enhance crop for OCR
                enhanced_crop = self.enhance_vehicle_crop_for_ocr(vehicle_crop)
                
                # Find potential license plate regions within the vehicle
                plate_regions = self.find_license_plate_regions(vehicle_crop)
                print(f"  Found {len(plate_regions)} potential license plate regions")
                
                vehicle_crop_info = {
                    'vehicle_id': i + 1,
                    'vehicle_bbox': bbox,
                    'vehicle_class': detection['class_name'],
                    'vehicle_confidence': detection['confidence'],
                    'license_plates': []
                }
                
                # Process the entire enhanced vehicle crop
                print(f"  Running OCR on enhanced vehicle crop...")
                ocr_result = self.lpr.extract_text_from_array(enhanced_crop, f"vehicle_{i+1}_enhanced")
                
                if not ocr_result.get("error") and ocr_result.get("cleaned_text"):
                    if (ocr_result['confidence'] >= self.ocr_confidence and 
                        ocr_result['is_valid_plate'] and 
                        len(ocr_result['cleaned_text']) >= 3):
                        
                        license_plate = {
                            'text': ocr_result['cleaned_text'],
                            'confidence': ocr_result['confidence'],
                            'method': 'enhanced_vehicle_crop',
                            'vehicle_bbox': bbox,
                            'raw_text': ocr_result['raw_text']
                        }
                        
                        all_license_plates.append(license_plate)
                        vehicle_crop_info['license_plates'].append(license_plate)
                        print(f"  ✅ License plate found: '{ocr_result['cleaned_text']}' (confidence: {ocr_result['confidence']}%)")
                
                # Also try enhanced OCR on individual license plate regions if found
                for j, (px, py, pw, ph) in enumerate(plate_regions):
                    plate_crop = vehicle_crop[py:py+ph, px:px+pw]
                    if plate_crop.size == 0:
                        continue
                    
                    print(f"  Running enhanced OCR on license plate region {j+1}...")
                    
                    # Use enhanced OCR specifically designed for license plates
                    debug_path = f"vehicle_{i+1}_region_{j+1}_debug.jpg" if True else None
                    enhanced_ocr_result = self.enhanced_ocr.extract_text_from_plate_region(
                        plate_crop, debug_path
                    )
                    
                    if not enhanced_ocr_result.get("error") and enhanced_ocr_result.get("text"):
                        if (enhanced_ocr_result['confidence'] >= self.ocr_confidence and 
                            enhanced_ocr_result.get('is_valid_plate', False) and 
                            len(enhanced_ocr_result['text']) >= 3):
                            
                            license_plate = {
                                'text': enhanced_ocr_result['text'],
                                'confidence': enhanced_ocr_result['confidence'],
                                'method': f'enhanced_ocr_region_{j+1}',
                                'vehicle_bbox': bbox,
                                'plate_region': {'x': px, 'y': py, 'w': pw, 'h': ph},
                                'raw_text': enhanced_ocr_result['raw_text'],
                                'enhancement_method': enhanced_ocr_result.get('enhancement_method', 'unknown'),
                                'total_attempts': enhanced_ocr_result.get('total_attempts', 0)
                            }
                            
                            # Avoid duplicates
                            if not any(lp['text'] == license_plate['text'] for lp in all_license_plates):
                                all_license_plates.append(license_plate)
                                vehicle_crop_info['license_plates'].append(license_plate)
                                print(f"  ✅ Enhanced OCR found license plate: '{enhanced_ocr_result['text']}' (confidence: {enhanced_ocr_result['confidence']:.1f}%)")
                        else:
                            print(f"  ⚠️  Enhanced OCR found text but low confidence: '{enhanced_ocr_result['text']}' ({enhanced_ocr_result['confidence']:.1f}%)")
                    else:
                        if enhanced_ocr_result.get('all_attempts_summary'):
                            print(f"  ❌ Enhanced OCR failed. Top attempts: {', '.join(enhanced_ocr_result['all_attempts_summary'][:3])}")
                        else:
                            print(f"  ❌ Enhanced OCR failed completely")
                
                if not vehicle_crop_info['license_plates']:
                    print(f"  ❌ No license plates detected for this vehicle")
                
                vehicle_crops_info.append(vehicle_crop_info)
            
            # Compile final results
            result = {
                "image_path": image_path,
                "image_size": vehicle_results['image_size'],
                "vehicles_detected": vehicle_results['vehicles_detected'],
                "license_plates_found": len(all_license_plates),
                "vehicle_detections": vehicle_results['detections'],
                "license_plates": all_license_plates,
                "vehicle_crops_analysis": vehicle_crops_info,
                "processing_summary": {
                    "vehicle_confidence_threshold": self.vehicle_confidence,
                    "ocr_confidence_threshold": self.ocr_confidence,
                    "vehicles_with_plates": len([v for v in vehicle_crops_info if v['license_plates']]),
                    "total_plate_detections": len(all_license_plates)
                }
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error in integrated pipeline: {str(e)}"}
    
    def create_visualization(self, image_path: str, results: Dict, output_path: str = None) -> str:
        """
        Create a visualization showing both vehicle detections and license plates.
        
        Args:
            image_path: Path to original image
            results: Results from process_image
            output_path: Path to save visualization
            
        Returns:
            Path to visualization image
        """
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_integrated_results{ext}"
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Draw vehicle bounding boxes
        for detection in results.get('vehicle_detections', []):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Draw vehicle box in blue
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Vehicle label
            vehicle_label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            cv2.putText(image, vehicle_label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw license plate information
        for i, lp in enumerate(results.get('license_plates', [])):
            bbox = lp['vehicle_bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Draw license plate text in green
            plate_label = f"PLATE: {lp['text']} ({lp['confidence']:.0f}%)"
            label_y = y2 + 25 + (i * 20)  # Stack multiple plates
            cv2.putText(image, plate_label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw a green border around vehicles with detected plates
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add summary text
        summary_text = f"Vehicles: {results.get('vehicles_detected', 0)} | License Plates: {results.get('license_plates_found', 0)}"
        cv2.putText(image, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    def create_detailed_visualization(self, image_path: str, results: Dict, output_path: str = None) -> str:
        """
        Create a detailed visualization showing vehicles, license plate regions, and results.
        
        Args:
            image_path: Path to original image
            results: Results from process_image
            output_path: Path to save visualization
            
        Returns:
            Path to visualization image
        """
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_detailed_analysis{ext}"
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Create a copy for drawing
        vis_image = image.copy()
        
        # Process each vehicle from the analysis
        for vehicle_info in results.get('vehicle_crops_analysis', []):
            vehicle_id = vehicle_info['vehicle_id']
            bbox = vehicle_info['vehicle_bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Draw vehicle bounding box
            if vehicle_info['license_plates']:
                # Green for vehicles with detected plates
                color = (0, 255, 0)
                thickness = 3
            else:
                # Blue for vehicles without plates
                color = (255, 0, 0)
                thickness = 2
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Vehicle label
            vehicle_label = f"V{vehicle_id}: {vehicle_info['vehicle_class']} ({vehicle_info['vehicle_confidence']:.2f})"
            cv2.putText(vis_image, vehicle_label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Extract vehicle crop to find license plate regions
            vehicle_crop = image[y1:y2, x1:x2]
            if vehicle_crop.size > 0:
                plate_regions = self.find_license_plate_regions(vehicle_crop)
                
                # Draw potential license plate regions
                for i, (px, py, pw, ph) in enumerate(plate_regions):
                    # Convert relative coordinates to absolute
                    abs_x1 = x1 + px
                    abs_y1 = y1 + py
                    abs_x2 = abs_x1 + pw
                    abs_y2 = abs_y1 + ph
                    
                    # Check if this region produced a successful OCR result
                    region_found_plate = False
                    for plate in vehicle_info['license_plates']:
                        if 'plate_region' in plate and f"license_plate_region_{i+1}" in plate['method']:
                            region_found_plate = True
                            break
                    
                    if region_found_plate:
                        # Yellow for regions that produced license plates
                        region_color = (0, 255, 255)
                        region_thickness = 2
                    else:
                        # Orange for potential regions that didn't produce results
                        region_color = (0, 165, 255)
                        region_thickness = 1
                    
                    cv2.rectangle(vis_image, (abs_x1, abs_y1), (abs_x2, abs_y2), region_color, region_thickness)
                    
                    # Small label for the region
                    region_label = f"R{i+1}"
                    cv2.putText(vis_image, region_label, (abs_x1, abs_y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, region_color, 1)
        
        # Add legend
        legend_y = 50
        cv2.putText(vis_image, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, "Green: Vehicle with license plate", (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, "Blue: Vehicle without license plate", (10, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(vis_image, "Yellow: License plate region (successful)", (10, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(vis_image, "Orange: Potential plate region", (10, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Add summary
        summary_y = image.shape[0] - 60
        summary_text = f"Vehicles: {results.get('vehicles_detected', 0)} | License Plates: {results.get('license_plates_found', 0)}"
        cv2.putText(vis_image, summary_text, (10, summary_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # List detected license plates
        if results.get('license_plates'):
            plates_text = "Detected: " + ", ".join([lp['text'] for lp in results['license_plates']])
            cv2.putText(vis_image, plates_text, (10, summary_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(output_path, vis_image)
        print(f"Detailed analysis visualization saved to: {output_path}")
        return output_path

    def create_vehicle_crops_visualization(self, image_path: str, results: Dict, output_dir: str = None) -> List[str]:
        """
        Create individual visualizations for each vehicle crop showing license plate regions.
        
        Args:
            image_path: Path to original image
            results: Results from process_image
            output_dir: Directory to save individual vehicle crops
            
        Returns:
            List of paths to saved vehicle crop visualizations
        """
        if output_dir is None:
            base_dir = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.join(base_dir, f"{base_name}_vehicle_crops")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            return []
        
        saved_paths = []
        
        for vehicle_info in results.get('vehicle_crops_analysis', []):
            vehicle_id = vehicle_info['vehicle_id']
            bbox = vehicle_info['vehicle_bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Extract vehicle crop
            vehicle_crop = original_image[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue
            
            # Create enhanced version for comparison
            enhanced_crop = self.enhance_vehicle_crop_for_ocr(vehicle_crop)
            enhanced_bgr = cv2.cvtColor(enhanced_crop, cv2.COLOR_GRAY2BGR)
            
            # Find license plate regions
            plate_regions = self.find_license_plate_regions(vehicle_crop)
            
            # Draw regions on both original and enhanced crops
            crop_with_regions = vehicle_crop.copy()
            enhanced_with_regions = enhanced_bgr.copy()
            
            for i, (px, py, pw, ph) in enumerate(plate_regions):
                # Check if this region produced a successful OCR result
                region_found_plate = False
                plate_text = ""
                for plate in vehicle_info['license_plates']:
                    if 'plate_region' in plate and f"license_plate_region_{i+1}" in plate['method']:
                        region_found_plate = True
                        plate_text = plate['text']
                        break
                
                if region_found_plate:
                    color = (0, 255, 255)  # Yellow for successful regions
                    thickness = 2
                else:
                    color = (0, 165, 255)  # Orange for potential regions
                    thickness = 1
                
                # Draw on original crop
                cv2.rectangle(crop_with_regions, (px, py), (px + pw, py + ph), color, thickness)
                cv2.putText(crop_with_regions, f"R{i+1}", (px, py - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw on enhanced crop
                cv2.rectangle(enhanced_with_regions, (px, py), (px + pw, py + ph), color, thickness)
                cv2.putText(enhanced_with_regions, f"R{i+1}", (px, py - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Add plate text if found
                if plate_text:
                    cv2.putText(crop_with_regions, plate_text, (px, py + ph + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(enhanced_with_regions, plate_text, (px, py + ph + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Resize for better visibility
            scale_factor = max(2.0, 300 / max(crop_with_regions.shape[:2]))
            new_size = (int(crop_with_regions.shape[1] * scale_factor), 
                       int(crop_with_regions.shape[0] * scale_factor))
            
            crop_resized = cv2.resize(crop_with_regions, new_size, interpolation=cv2.INTER_CUBIC)
            enhanced_resized = cv2.resize(enhanced_with_regions, new_size, interpolation=cv2.INTER_CUBIC)
            
            # Create side-by-side comparison
            combined = np.hstack([crop_resized, enhanced_resized])
            
            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Enhanced", (crop_resized.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add vehicle info
            info_text = f"Vehicle {vehicle_id}: {vehicle_info['vehicle_class']} ({vehicle_info['vehicle_confidence']:.2f})"
            cv2.putText(combined, info_text, (10, combined.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save individual vehicle visualization
            vehicle_path = os.path.join(output_dir, f"vehicle_{vehicle_id:02d}_{vehicle_info['vehicle_class']}.jpg")
            cv2.imwrite(vehicle_path, combined)
            saved_paths.append(vehicle_path)
        
        print(f"Vehicle crop visualizations saved to: {output_dir}")
        print(f"Created {len(saved_paths)} individual vehicle analysis images")
        return saved_paths
