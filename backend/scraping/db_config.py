import psycopg2
import os
from contextlib import contextmanager

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'formula_motorsport'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'port': os.getenv('DB_PORT', 5433)
}

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def create_tables():
    """Create database tables if they don't exist"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Championships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS championships (
                id SERIAL PRIMARY KEY,
                year INTEGER NOT NULL,
                formula INTEGER NOT NULL,
                series_type VARCHAR(50) DEFAULT 'main',
                championship_type VARCHAR(20) NOT NULL,
                position INTEGER,
                driver_team VARCHAR(255),
                car_number VARCHAR(10),
                points DECIMAL(5,1),
                race_results JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(year, formula, series_type, championship_type, position)
            )
        ''')
        
        # Entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                id SERIAL PRIMARY KEY,
                year INTEGER NOT NULL,
                formula INTEGER NOT NULL,
                series_type VARCHAR(50) DEFAULT 'main',
                entrant VARCHAR(255),
                car_number VARCHAR(10),
                driver VARCHAR(255),
                chassis VARCHAR(100),
                engine VARCHAR(100),
                rounds VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Qualifying table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qualifying (
                id SERIAL PRIMARY KEY,
                year INTEGER NOT NULL,
                formula INTEGER NOT NULL,
                round_number INTEGER NOT NULL,
                position VARCHAR(10),
                car_number VARCHAR(10),
                driver VARCHAR(255),
                team VARCHAR(255),
                time VARCHAR(20),
                qualifying_type VARCHAR(50),
                url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Driver profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS driver_profiles (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                date_of_birth DATE,
                nationality VARCHAR(100),
                wiki_url VARCHAR(500),
                scraped BOOLEAN DEFAULT FALSE,
                scraped_date TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print("Database tables created successfully")

if __name__ == "__main__":
    create_tables()