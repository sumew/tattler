import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# --- Configuration ---
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "tattler_db")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# --- Main Reporting Logic ---

def find_long_parked_vehicles(engine, hours_threshold=2):
    """
    Finds vehicles that have been parked for longer than the threshold.
    This is a simplified query. A more robust implementation would check
    that the sightings are all within a small geographic radius.
    """
    print(f"Searching for vehicles parked for more than {hours_threshold} hours...")
    
    query = text(f"""
        SELECT 
            license_plate,
            MIN(timestamp) as first_seen,
            MAX(timestamp) as last_seen,
            MAX(timestamp) - MIN(timestamp) as duration
        FROM sightings
        GROUP BY license_plate
        HAVING MAX(timestamp) - MIN(timestamp) >= INTERVAL '{hours_threshold} hours'
        ORDER BY duration DESC;
    """)
    
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    
    return df

def main():
    print("Starting Reporting Engine...")
    engine = create_engine(DATABASE_URL)
    
    # --- Generate Overstay Report ---
    overstay_report = find_long_parked_vehicles(engine, hours_threshold=2)
    
    if not overstay_report.empty:
        report_filename = f"overstay_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        overstay_report.to_csv(report_filename, index=False)
        print(f"Generated overstay report: {report_filename}")
        print(overstay_report)
    else:
        print("No vehicles found exceeding the parking duration threshold.")

if __name__ == "__main__":
    main()
