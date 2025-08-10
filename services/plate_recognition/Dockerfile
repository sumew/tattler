FROM python:3.9-slim

WORKDIR /app

# Install OpenCV dependencies and other system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories so docker-compose volumes mount correctly
RUN mkdir -p /app/vehicle_crops
RUN mkdir -p /app/plate_crops
RUN mkdir -p /app/results

CMD ["python", "main.py"]
