FROM python:3.9-slim

WORKDIR /app

# Install Tesseract OCR Engine
RUN apt-get update && apt-get install -y tesseract-ocr

COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Add pytesseract if you plan to use it
RUN pip install pytesseract

COPY . .

CMD ["python", "main.py"]
