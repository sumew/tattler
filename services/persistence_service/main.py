import redis
import os
import json
import time
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import POINT
from geoalchemy2 import Geography
from datetime import datetime

# --- Configuration ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
PLATE_SIGHTINGS_CHANNEL = "plate_sightings"

DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "tattler_db")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# --- Database Setup ---
Base = declarative_base()

class Sighting(Base):
    __tablename__ = 'sightings'
    id = Column(Integer, primary_key=True)
    license_plate = Column(String(20), nullable=False)
    location = Column(Geography('POINT', srid=4326), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

def get_db_session():
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine) # Create table if it doesn't exist
    Session = sessionmaker(bind=engine)
    return Session()

# --- Main Application Logic ---

def main():
    print("Starting Persistence Service...")
    
    # Wait for DB to be ready
    time.sleep(10) 
    db_session = get_db_session()

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    pubsub = redis_client.pubsub()
    pubsub.subscribe(PLATE_SIGHTINGS_CHANNEL)
    print(f"Subscribed to Redis channel: {PLATE_SIGHTINGS_CHANNEL}")

    for message in pubsub.listen():
        if message['type'] == 'message':
            print("Received a new plate sighting.")
            data = json.loads(message['data'])
            
            plate = data['license_plate']
            loc = data['location']
            point = f"POINT({loc['lon']} {loc['lat']})"

            new_sighting = Sighting(
                license_plate=plate,
                location=point,
                timestamp=datetime.fromtimestamp(data['timestamp'])
            )
            
            db_session.add(new_sighting)
            db_session.commit()
            print(f"Saved sighting for plate {plate} to the database.")

if __name__ == "__main__":
    main()
