import json
from .db_config import get_db_connection
from .scraping_utils import remove_citations

def save_championship_to_db(data, year, formula, championship_type, series_type="main"):
    """Save championship standings to database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Clear existing data for this championship
        cursor.execute('''
            DELETE FROM championships 
            WHERE year = %s AND formula = %s AND championship_type = %s AND series_type = %s
        ''', (year, formula, championship_type, series_type))
        
        for row in data:
            if len(row) < 3:  # Skip incomplete rows
                continue
                
            position = row[0] if row[0] and row[0].isdigit() else None
            driver_team = remove_citations(row[1]) if len(row) > 1 else None
            points = None
            
            # Extract points (usually the last column)
            if row[-1] and row[-1].replace('.', '').replace('-', '').isdigit():
                points = float(row[-1])
            
            # Store race results as JSON (columns between driver_team and points)
            race_results = {}
            if len(row) > 3:  # Has race result columns
                race_columns = row[2:-1]  # Everything between driver_team and points
                for i, result in enumerate(race_columns, 1):
                    if result.strip():
                        race_results[f'race_{i}'] = result.strip()
            
            cursor.execute('''
                INSERT INTO championships 
                (year, formula, series_type, championship_type, position, driver_team, points, race_results)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (year, formula, series_type, championship_type, position) 
                DO UPDATE SET 
                    driver_team = EXCLUDED.driver_team,
                    points = EXCLUDED.points,
                    race_results = EXCLUDED.race_results
            ''', (year, formula, series_type, championship_type, position, driver_team, points, json.dumps(race_results)))
        
        conn.commit()
        print(f"Saved {championship_type} championship data for F{formula} {year} ({series_type})")

def save_entries_to_db(data, headers, year, formula, series_type="main"):
    """Save entries data to database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('''
            DELETE FROM entries 
            WHERE year = %s AND formula = %s AND series_type = %s
        ''', (year, formula, series_type))
        
        # Map headers to column names
        header_mapping = {
            'entrant': ['entrant', 'team', 'constructor'],
            'car_number': ['no.', 'no', 'number', 'car'],
            'driver': ['driver', 'drivers', 'driver name'],
            'chassis': ['chassis', 'car'],
            'engine': ['engine', 'engines'],
            'rounds': ['rounds', 'round', 'races']
        }
        
        # Find column indices
        col_indices = {}
        for col_name, possible_headers in header_mapping.items():
            for i, header in enumerate(headers):
                if header.lower() in [h.lower() for h in possible_headers]:
                    col_indices[col_name] = i
                    break
        
        for row in data:
            if len(row) < 3:  # Skip incomplete rows
                continue
                
            entrant = remove_citations(row[col_indices.get('entrant', 0)]) if col_indices.get('entrant') is not None else None
            car_number = row[col_indices.get('car_number', 1)] if col_indices.get('car_number') is not None else None
            driver = remove_citations(row[col_indices.get('driver', 2)]) if col_indices.get('driver') is not None else None
            chassis = row[col_indices.get('chassis')] if col_indices.get('chassis') is not None else None
            engine = row[col_indices.get('engine')] if col_indices.get('engine') is not None else None
            rounds = row[col_indices.get('rounds')] if col_indices.get('rounds') is not None else None
            
            cursor.execute('''
                INSERT INTO entries 
                (year, formula, series_type, entrant, car_number, driver, chassis, engine, rounds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (year, formula, series_type, entrant, car_number, driver, chassis, engine, rounds))
        
        conn.commit()
        print(f"Saved entries data for F{formula} {year} ({series_type})")
