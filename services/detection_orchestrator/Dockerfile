FROM python:3.9-slim

WORKDIR /app

COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directory so docker-compose volumes mount correctly
RUN mkdir -p /app/raw_frames

CMD ["python", "main.py"]
