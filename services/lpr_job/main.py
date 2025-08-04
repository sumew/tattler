import redis
import os
import json
import time
from PIL import Image
import io

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
PLATE_SIGHTINGS_CHANNEL = "plate_sightings"
INPUT_DIR = "detected_frames"
PROCESSED_DIR = os.path.join(INPUT_DIR, "processed")

# --- Main Application Logic ---

def get_gps_from_exif(image_path: str) -> dict | None:
    """
    Placeholder function to extract GPS data from image EXIF.
    A real implementation would use a library like `piexif`.
    """
    return {"lat": 40.7128, "lon": -74.0060}

def recognize_license_plate(image_path: str) -> str | None:
    """
    Recognizes a license plate from an image file.
    This is a placeholder. A real implementation would use Tesseract.
    """
    try:
        print(f"Processing image with OCR: {image_path}")
        time.sleep(1) # Simulate work
        
        import random
        if random.random() > 0.3: # 70% chance of success
            plate_number = f"GEM-{random.randint(100, 999)}"
            print(f"Successfully recognized plate: {plate_number}")
            return plate_number
        else:
            print("OCR failed to recognize a plate.")
            return None
            
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return None

def main():
    print("Starting LPR Processor Job...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    image_files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
    
    print(f"Found {len(image_files)} images to process.")

    for image_file in image_files:
        image_path = os.path.join(INPUT_DIR, image_file)
        
        plate_number = recognize_license_plate(image_path)

        if plate_number:
            gps_location = get_gps_from_exif(image_path)
            
            if gps_location:
                sighting_data = {
                    "license_plate": plate_number,
                    "location": gps_location,
                    "timestamp": time.time()
                }
                redis_client.publish(PLATE_SIGHTINGS_CHANNEL, json.dumps(sighting_data))
                print(f"Published sighting for plate {plate_number} to Redis.")
            else:
                print(f"Could not find GPS data for image {image_path}")
        
        # Move processed images to an archive folder
        try:
            os.rename(image_path, os.path.join(PROCESSED_DIR, image_file))
        except OSError as e:
            print(f"Error moving file {image_path}: {e}")

    print("LPR Processor Job finished.")

if __name__ == "__main__":
    main()