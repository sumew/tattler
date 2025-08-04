import os
import subprocess
import sys

# --- Configuration ---
VIDEO_DIR = "videos"
RAW_FRAME_DIR = "raw_frames"

def main():
    print("Starting FFmpeg Frame Extraction Job...")

    # --- Get video file from command-line arguments ---
    if len(sys.argv) < 2:
        print("Error: No video file specified.")
        print("Usage: python main.py <video_filename>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    video_path = os.path.join(VIDEO_DIR, video_file)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    if not os.path.exists(RAW_FRAME_DIR):
        os.makedirs(RAW_FRAME_DIR)

    # --- Run FFmpeg on the specified file ---
    output_pattern = os.path.join(RAW_FRAME_DIR, f"{os.path.splitext(video_file)[0]}_frame_%05d.jpg")
    
    print(f"Extracting frames from {video_path}...")
    
    # Use ffmpeg to extract frames. -q:v 2 sets the quality to high.
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '2',
        output_pattern
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully extracted frames to {RAW_FRAME_DIR}")
        # TODO: Move the processed video to an archive directory
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {video_file}:")
        print(e.stderr)

    print("FFmpeg Frame Extraction Job finished.")

if __name__ == "__main__":
    main()