FROM python:3.9-slim

WORKDIR /app

# Install Tesseract OCR Engine
RUN apt-get update && apt-get install -y tesseract-ocr

COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Add pytesseract and piexif for a real implementation
RUN pip install pytesseract piexif

COPY . .

# Create directory so docker-compose volumes mount correctly
RUN mkdir -p /app/detected_frames

CMD ["python", "main.py"]