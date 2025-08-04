import cv2
import os
import shutil
import redis
import json

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
DETECT_CHANNEL = "frames_to_detect"

RAW_FRAME_DIR = "raw_frames"
DETECTED_FRAME_DIR = "detected_frames"

CAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_car.xml'
PLATE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'

class DetectionService:
    def __init__(self):
        print("Initializing Detection Service...")
        self._load_models()
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(DETECT_CHANNEL)
        print("Initialization complete. Models loaded.")

    def _load_models(self):
        """Loads the detection models into memory."""
        self.car_cascade = cv2.CascadeClassifier(CAR_CASCADE_PATH)
        self.plate_cascade = cv2.CascadeClassifier(PLATE_CASCADE_PATH)
        if self.car_cascade.empty() or self.plate_cascade.empty():
            raise IOError("Could not load Haar Cascade models.")

    def process_image(self, image_path):
        """Runs the detection logic on a single image file."""
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return

        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}, deleting.")
                os.remove(image_path)
                return

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = self.car_cascade.detectMultiScale(gray, 1.1, 1)
            plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)

            if len(cars) > 0 and len(plates) > 0:
                print(f"SUCCESS: Car/plate combo found in {os.path.basename(image_path)}.")
                if not os.path.exists(DETECTED_FRAME_DIR):
                    os.makedirs(DETECTED_FRAME_DIR)
                destination_path = os.path.join(DETECTED_FRAME_DIR, os.path.basename(image_path))
                shutil.move(image_path, destination_path)
            else:
                print(f"INFO: No car/plate combo in {os.path.basename(image_path)}. Deleting.")
                os.remove(image_path)
        except Exception as e:
            print(f"An error occurred processing {image_path}: {e}")

    def start(self):
        """Starts the service and listens for messages on the Redis channel."""
        print(f"Listening for messages on Redis channel: '{DETECT_CHANNEL}'...")
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                image_path = data.get('image_path')
                if image_path:
                    print(f"--- Received message to process: {image_path} ---")
                    self.process_image(image_path)

if __name__ == "__main__":
    service = DetectionService()
    service.start()
