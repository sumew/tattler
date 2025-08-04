import redis
import os
import json

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
DETECT_CHANNEL = "frames_to_detect"
RAW_FRAME_DIR = "raw_frames"

def main():
    print("Starting Detection Orchestrator Job...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    if not os.path.isdir(RAW_FRAME_DIR):
        print(f"Error: Directory not found: {RAW_FRAME_DIR}")
        return

    image_files = [f for f in os.listdir(RAW_FRAME_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} raw frames to queue for detection.")

    queued_count = 0
    for image_file in image_files:
        # The path must be the one accessible from within the service container
        image_path_in_container = os.path.join("/app/raw_frames", image_file)
        
        message = {
            "image_path": image_path_in_container
        }
        
        redis_client.publish(DETECT_CHANNEL, json.dumps(message))
        print(f"Queued {image_path_in_container} for processing.")
        queued_count += 1

    print("-----------------------------------------------------")
    print(f"Detection Orchestrator finished. Queued {queued_count} frames.")

if __name__ == "__main__":
    main()
