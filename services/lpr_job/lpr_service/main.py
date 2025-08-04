import redis
import os
import json
import time
from PIL import Image
import io

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
FRAME_PROCESS_CHANNEL = "frames_to_process"
PLATE_SIGHTINGS_CHANNEL = "plate_sightings"

# --- Main Application Logic ---

def recognize_license_plate(image_bytes: bytes) -> str | None:
    """
    Recognizes a license plate from image bytes.
    This is a placeholder. A real implementation would use Tesseract,
    a cloud API, or a dedicated LPR model.
    """
    try:
        # Simulate OCR processing
        print("Processing image with OCR...")
        image = Image.open(io.BytesIO(image_bytes))
        
        # TODO: Implement actual OCR logic here.
        # For now, we'll return a dummy plate number.
        time.sleep(1) # Simulate work
        
        # In a real scenario, you might not always find a plate
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
    print("Starting LPR Service...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    pubsub = redis_client.pubsub()
    pubsub.subscribe(FRAME_PROCESS_CHANNEL)
    print(f"Subscribed to Redis channel: {FRAME_PROCESS_CHANNEL}")

    for message in pubsub.listen():
        if message['type'] == 'message':
            print("Received a new frame to process.")
            data = json.loads(message['data'])
            
            image_bytes = data['image_bytes'].encode('latin-1') # Decode from transport format
            gps = data['gps']

            plate_number = recognize_license_plate(image_bytes)

            if plate_number:
                sighting_data = {
                    "license_plate": plate_number,
                    "location": gps,
                    "timestamp": time.time()
                }
                # Publish the result
                redis_client.publish(PLATE_SIGHTINGS_CHANNEL, json.dumps(sighting_data))
                print(f"Published sighting for plate {plate_number} to Redis.")


if __name__ == "__main__":
    main()
