FROM python:3.9-slim

WORKDIR /app

# Install build dependencies for psycopg2 and geoalchemy2
RUN apt-get update && apt-get install -y build-essential libpq-dev

COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Add geoalchemy2 for PostGIS support
RUN pip install geoalchemy2

COPY . .

CMD ["python", "main.py"]
