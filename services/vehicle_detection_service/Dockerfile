FROM python:3.9-slim

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories so docker-compose volumes mount correctly
RUN mkdir -p /app/raw_frames
RUN mkdir -p /app/detected_frames

CMD ["python", "main.py"]