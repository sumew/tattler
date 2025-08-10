# FFmpeg Extraction Service

High-performance video frame extraction service using FFmpeg. This service efficiently extracts frames from video files for further processing by other services in the Tattler pipeline.

## Features

- **High Performance**: Uses FFmpeg for optimized video processing
- **Flexible Extraction**: Extract frames at custom intervals or specific timestamps
- **Multiple Formats**: Supports all major video formats (MP4, AVI, MOV, MKV, etc.)
- **Quality Control**: Configurable output quality and resolution
- **Batch Processing**: Process multiple videos efficiently
- **Memory Efficient**: Streams processing without loading entire videos

## Installation

```bash
# Install from the tattler root directory
pip install -e .
pip install -e ./services/ffmpeg_extraction_job

# Install FFmpeg (required dependency)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Quick Start

```python
from tattler_ffmpeg.main import extract_frames

# Extract frames every 5 seconds
extract_frames(
    input_video="parking_lot_video.mp4",
    output_dir="extracted_frames/",
    interval=5.0
)

# Extract specific number of frames
extract_frames(
    input_video="drone_footage.mp4",
    output_dir="frames/",
    max_frames=100
)
```

## API Reference

### extract_frames()

Extract frames from a video file using FFmpeg.

```python
def extract_frames(
    input_video: str,
    output_dir: str,
    interval: float = 1.0,
    max_frames: Optional[int] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    quality: int = 2,
    resolution: Optional[str] = None,
    format: str = "jpg"
) -> Dict[str, Any]:
```

**Parameters:**
- `input_video` (str): Path to input video file
- `output_dir` (str): Directory to save extracted frames
- `interval` (float): Time interval between frames in seconds. Default: 1.0
- `max_frames` (int, optional): Maximum number of frames to extract
- `start_time` (float): Start extraction from this time (seconds). Default: 0.0
- `end_time` (float, optional): Stop extraction at this time (seconds)
- `quality` (int): JPEG quality (1-31, lower is better). Default: 2
- `resolution` (str, optional): Output resolution (e.g., "1920x1080", "1280x720")
- `format` (str): Output format ("jpg", "png"). Default: "jpg"

**Returns:**
- `Dict[str, Any]`: Extraction results and statistics

**Example Response:**
```python
{
    "success": True,
    "frames_extracted": 120,
    "output_directory": "extracted_frames/",
    "video_duration": 300.5,
    "extraction_time": 15.2,
    "average_fps": 7.9,
    "output_files": ["frame_001.jpg", "frame_002.jpg", ...]
}
```

## Usage Examples

### Basic Frame Extraction
```python
from tattler_ffmpeg.main import extract_frames

# Extract one frame per second
result = extract_frames(
    input_video="surveillance.mp4",
    output_dir="frames/",
    interval=1.0
)

print(f"Extracted {result['frames_extracted']} frames")
```

### High-Quality Extraction
```python
# Extract high-quality frames every 30 seconds
result = extract_frames(
    input_video="drone_footage.mp4",
    output_dir="high_quality_frames/",
    interval=30.0,
    quality=1,  # Highest quality
    resolution="1920x1080",
    format="png"
)
```

### Time-Range Extraction
```python
# Extract frames from 2 minutes to 5 minutes
result = extract_frames(
    input_video="long_video.mp4",
    output_dir="segment_frames/",
    start_time=120.0,  # 2 minutes
    end_time=300.0,    # 5 minutes
    interval=2.0       # Every 2 seconds
)
```

### Limited Frame Count
```python
# Extract exactly 50 frames evenly distributed
result = extract_frames(
    input_video="sample.mp4",
    output_dir="sample_frames/",
    max_frames=50
)
```

## Command Line Usage

The service can also be used as a command-line tool:

```bash
# Basic extraction
python -m tattler_ffmpeg.main --input video.mp4 --output frames/ --interval 5

# Advanced options
python -m tattler_ffmpeg.main \
    --input surveillance.mp4 \
    --output extracted/ \
    --interval 2.0 \
    --start-time 60 \
    --end-time 300 \
    --quality 1 \
    --resolution 1280x720 \
    --max-frames 100
```

## Integration with Other Services

### With Vehicle Detection
```python
from tattler_ffmpeg.main import extract_frames
from tattler_vehicle_detection.detector import VehicleDetector

# Step 1: Extract frames from video
frame_result = extract_frames(
    input_video="parking_lot.mp4",
    output_dir="frames/",
    interval=5.0
)

# Step 2: Detect vehicles in each frame
detector = VehicleDetector()
for frame_file in frame_result['output_files']:
    detections = detector.detect_vehicles(f"frames/{frame_file}")
    if detections['has_vehicles']:
        print(f"Found {detections['vehicles_detected']} vehicles in {frame_file}")
```

### With Plate Recognition
```python
from tattler_ffmpeg.main import extract_frames
from tattler_plate_recognition.recognizer import PlateRecognizer

# Extract frames and detect license plates
frame_result = extract_frames("traffic.mp4", "frames/", interval=3.0)

recognizer = PlateRecognizer()
for frame_file in frame_result['output_files']:
    detections = recognizer.detect_plate(f"frames/{frame_file}")
    if detections:
        print(f"Found {len(detections)} license plates in {frame_file}")
```

## Performance Optimization

### Speed Tips
1. **Use appropriate intervals**: Don't extract more frames than needed
2. **Lower quality for processing**: Use quality=10-15 for AI processing
3. **Resize videos**: Use `resolution` parameter to reduce frame size
4. **Batch processing**: Process multiple videos in sequence

### Memory Management
```python
# Process large videos in segments
def process_large_video(video_path, segment_duration=300):  # 5-minute segments
    video_info = get_video_info(video_path)  # Custom function
    total_duration = video_info['duration']
    
    for start in range(0, int(total_duration), segment_duration):
        end = min(start + segment_duration, total_duration)
        
        extract_frames(
            input_video=video_path,
            output_dir=f"frames_segment_{start//segment_duration}/",
            start_time=start,
            end_time=end,
            interval=2.0
        )
```

## Docker Deployment

```dockerfile
FROM python:3.10-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared utilities
COPY src/tattler/ ./src/tattler/
COPY pyproject.toml ./
RUN pip install -e .

# Copy service
COPY services/ffmpeg_extraction_job/ ./
RUN pip install -e .

# Create volume for video processing
VOLUME ["/app/videos", "/app/frames"]

CMD ["python", "-m", "tattler_ffmpeg.main"]
```

## Supported Video Formats

- **MP4** (.mp4) - Most common, recommended
- **AVI** (.avi) - Legacy format
- **MOV** (.mov) - Apple QuickTime
- **MKV** (.mkv) - Matroska container
- **WMV** (.wmv) - Windows Media Video
- **FLV** (.flv) - Flash Video
- **WEBM** (.webm) - Web-optimized format
- **M4V** (.m4v) - iTunes video format

## Output Formats

- **JPEG** (.jpg) - Default, good compression
- **PNG** (.png) - Lossless, larger files

## Quality Settings

| Quality | Description | Use Case |
|---------|-------------|----------|
| 1-5     | Highest quality | Final output, archival |
| 6-15    | High quality | AI processing, analysis |
| 16-25   | Medium quality | Preview, thumbnails |
| 26-31   | Lower quality | Quick processing |

## Troubleshooting

### FFmpeg Not Found
```
Error: FFmpeg not found in system PATH
```
**Solution**: Install FFmpeg using your system's package manager

### Video Format Not Supported
```
Error: Could not open video file
```
**Solution**: 
- Check if file exists and is readable
- Verify video format is supported
- Try converting video to MP4 format

### Out of Disk Space
```
Error: No space left on device
```
**Solution**:
- Check available disk space
- Use lower quality settings
- Extract fewer frames or shorter segments
- Clean up old extracted frames

### Memory Issues
```
Error: Out of memory
```
**Solution**:
- Process videos in smaller segments
- Use lower resolution output
- Increase system swap space
- Process one video at a time

## Configuration

Create a configuration file for default settings:

```python
# config.py
DEFAULT_CONFIG = {
    "interval": 2.0,
    "quality": 5,
    "format": "jpg",
    "resolution": "1280x720",
    "max_frames": 1000
}
```

## Monitoring and Logging

The service provides detailed logging for monitoring extraction progress:

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

result = extract_frames("video.mp4", "frames/", interval=1.0)
# Logs will show:
# INFO: Starting frame extraction from video.mp4
# INFO: Extracted frame 1/120
# INFO: Extraction complete: 120 frames in 15.2 seconds
```

## Best Practices

1. **Test with small videos first** to verify settings
2. **Use consistent naming** for output directories
3. **Monitor disk space** when processing large videos
4. **Clean up old frames** regularly to save space
5. **Use appropriate quality** for your use case
6. **Consider video resolution** vs processing speed trade-offs
