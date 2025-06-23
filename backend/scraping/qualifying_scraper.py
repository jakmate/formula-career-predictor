import re
import requests
from bs4 import BeautifulSoup

from .db_config import get_db_connection
from .scraping_utils import remove_citations


def time_to_seconds(time_str):
    """Convert time string to seconds for calculations"""
    if not time_str or time_str in ['—', '-', '']:
        return None
    
    # Remove any extra whitespace and prefixes
    time_str = time_str.strip().replace('+', '').replace('s', '')
    
    # Handle formats like "1:46.559" or "0.157"
    if ':' in time_str:
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        try:
            return float(time_str)
        except ValueError:
            return None


def seconds_to_time(seconds):
    """Convert seconds back to time string format"""
    if seconds is None:
        return None
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes > 0:
        return f"{minutes}:{secs:06.3f}"
    else:
        return f"{secs:.3f}"


def preprocess_qualifying_data(headers, data):
    """Preprocess qualifying data based on column structure"""
    processed_data = []
    
    # Normalize headers for easier matching
    norm_headers = [h.lower().strip() for h in headers]
    
    # Find column indices
    pos_col = next((i for i, h in enumerate(norm_headers) if h in ['pos', 'pos.']), 0)
    no_col = next((i for i, h in enumerate(norm_headers) if h in ['no', 'no.']), 1)
    driver_col = next((i for i, h in enumerate(norm_headers) if h in ['driver', 'name']), 2)
    team_col = next((i for i, h in enumerate(norm_headers) if h in ['team', 'constructor', 'entrant']), 3)
    
    # Identify the qualifying format
    qualifying_format = identify_qualifying_format(norm_headers)
    
    for row_idx, row in enumerate(data):
        if len(row) < 3:
            continue
        
        # Handle position column edge cases
        position = row[pos_col] if len(row) > pos_col else None
        if position in ['-', '–', '—', '']:
            position = str(row_idx + 1)  # Use table order
        elif position and position.startswith('DSQ'):
            position = position  # Keep DSQ1, DSQ2, etc.
        elif position in ['NC', 'NC3', 'EX', 'WD']:
            position = position  # Keep special statuses
            
        processed_row = {
            'position': position,
            'car_number': row[no_col] if len(row) > no_col else None,
            'driver': remove_citations(row[driver_col]) if len(row) > driver_col else None,
            'team': row[team_col] if len(row) > team_col else None,
            'time': None,
            'grid': None
        }
        
        # Process based on format
        if qualifying_format == 'time_gap':
            processed_row.update(process_time_gap_format(headers, row, data))
        elif qualifying_format == 'time_gap_combined':
            processed_row.update(process_time_gap_combined_format(headers, row, data))
        elif qualifying_format == 'q1_q2':
            processed_row.update(process_q1_q2_format(headers, row))
        elif qualifying_format == 'r1_r2':
            processed_row.update(process_r1_r2_format(headers, row))
        elif qualifying_format == 'grid_sr_fr':
            processed_row.update(process_grid_sr_fr_format(headers, row))
        else:  # standard format
            processed_row.update(process_standard_format(headers, row))
            
        processed_data.append(processed_row)
    
    return processed_data


def identify_qualifying_format(norm_headers):
    """Identify the qualifying data format based on headers"""
    header_str = ' '.join(norm_headers)
    
    if 'q1 time' in header_str and 'q2 time' in header_str:
        return 'q1_q2'
    elif 'r1' in header_str and 'r2' in header_str:
        return 'r1_r2'
    elif 'gridsr' in header_str and 'gridfr' in header_str:
        return 'grid_sr_fr'
    elif 'time/gap' in header_str:
        return 'time_gap_combined'
    elif 'time' in header_str and 'gap' in header_str:
        return 'time_gap'
    else:
        return 'standard'


def process_standard_format(headers, row):
    """Process standard Time,Grid format"""
    norm_headers = [h.lower().strip() for h in headers]
    
    time_col = next((i for i, h in enumerate(norm_headers) if 'time' in h), None)
    grid_col = next((i for i, h in enumerate(norm_headers) if 'grid' in h), None)
    
    result = {}
    
    if time_col is not None and len(row) > time_col:
        result['time'] = row[time_col]
    
    if grid_col is not None and len(row) > grid_col:
        result['grid'] = row[grid_col]
    
    return result


def process_time_gap_format(headers, row, all_data):
    """Process Time,Gap,Grid format - add gap to fastest time in the group"""
    norm_headers = [h.lower().strip() for h in headers]
    
    time_col = next((i for i, h in enumerate(norm_headers) if 'time' in h), None)
    gap_col = next((i for i, h in enumerate(norm_headers) if 'gap' in h), None)
    grid_col = next((i for i, h in enumerate(norm_headers) if 'grid' in h), None)
    
    result = {}
    
    if time_col is not None and len(row) > time_col:
        base_time = row[time_col]
        gap = row[gap_col] if gap_col is not None and len(row) > gap_col else None
        
        if gap and gap not in ['—', '-', '']:
            # Find the fastest time in this group (first row with actual time)
            fastest_time = None
            for data_row in all_data:
                if len(data_row) > time_col and data_row[time_col] and ':' in data_row[time_col]:
                    fastest_time = data_row[time_col]
                    break
            
            if fastest_time:
                # Parse gap (e.g., "+ 0.157 s" or "+0.485")
                gap_clean = re.sub(r'[+\s]|s$', '', gap)
                try:
                    gap_seconds = float(gap_clean)
                    base_seconds = time_to_seconds(fastest_time)
                    if base_seconds is not None:
                        total_seconds = base_seconds + gap_seconds
                        result['time'] = seconds_to_time(total_seconds)
                    else:
                        result['time'] = base_time
                except ValueError:
                    result['time'] = base_time
            else:
                result['time'] = base_time
        else:
            result['time'] = base_time
    
    if grid_col is not None and len(row) > grid_col:
        result['grid'] = row[grid_col]
    
    return result


def process_time_gap_combined_format(headers, row, all_data):
    """Process Time/Gap column - fastest has time, others have gap"""
    norm_headers = [h.lower().strip() for h in headers]
    
    time_gap_col = next((i for i, h in enumerate(norm_headers) if 'time/gap' in h), None)
    grid_col = next((i for i, h in enumerate(norm_headers) if 'grid' in h), None)
    
    result = {}
    
    if time_gap_col is not None and len(row) > time_gap_col:
        time_gap_value = row[time_gap_col]
        
        # Check if this is the fastest time (contains ":")
        if ':' in time_gap_value:
            result['time'] = time_gap_value
        else:
            # This is a gap, find the fastest time from first row
            fastest_time = None
            for data_row in all_data:
                if len(data_row) > time_gap_col and ':' in data_row[time_gap_col]:
                    fastest_time = data_row[time_gap_col]
                    break
            
            if fastest_time:
                gap_clean = re.sub(r'[+\s]', '', time_gap_value)
                try:
                    gap_seconds = float(gap_clean)
                    base_seconds = time_to_seconds(fastest_time)
                    if base_seconds is not None:
                        total_seconds = base_seconds + gap_seconds
                        result['time'] = seconds_to_time(total_seconds)
                    else:
                        result['time'] = time_gap_value
                except ValueError:
                    result['time'] = time_gap_value
            else:
                result['time'] = time_gap_value
    
    if grid_col is not None and len(row) > grid_col:
        result['grid'] = row[grid_col]
    
    return result


def process_q1_q2_format(headers, row):
    """Process Q1 Time,Rank,Q2 Time,Rank,Gap,Grid format - use Q2 time + gap"""
    norm_headers = [h.lower().strip() for h in headers]
    
    q2_time_col = next((i for i, h in enumerate(norm_headers) if 'q2 time' in h), None)
    gap_col = next((i for i, h in enumerate(norm_headers) if 'gap' in h), None)
    grid_col = next((i for i, h in enumerate(norm_headers) if 'grid' in h), None)
    
    result = {}
    
    if q2_time_col is not None and len(row) > q2_time_col:
        q2_time = row[q2_time_col]
        gap = row[gap_col] if gap_col is not None and len(row) > gap_col else None
        
        if gap and gap not in ['—', '-', '']:
            gap_clean = re.sub(r'[+\s]', '', gap)
            try:
                gap_seconds = float(gap_clean)
                base_seconds = time_to_seconds(q2_time)
                if base_seconds is not None:
                    total_seconds = base_seconds + gap_seconds
                    result['time'] = seconds_to_time(total_seconds)
                else:
                    result['time'] = q2_time
            except ValueError:
                result['time'] = q2_time
        else:
            result['time'] = q2_time
    
    if grid_col is not None and len(row) > grid_col:
        result['grid'] = row[grid_col]
    
    return result


def process_r1_r2_format(headers, row):
    """Process R1,R2 format - use R2 as grid position"""
    norm_headers = [h.lower().strip() for h in headers]
    
    time_gap_col = next((i for i, h in enumerate(norm_headers) if 'time/gap' in h), None)
    r2_col = next((i for i, h in enumerate(norm_headers) if 'r2' in h), None)
    
    result = {}
    
    if time_gap_col is not None and len(row) > time_gap_col:
        result['time'] = row[time_gap_col]
    
    if r2_col is not None and len(row) > r2_col:
        result['grid'] = row[r2_col]
    
    return result


def process_grid_sr_fr_format(headers, row):
    """Process GridSR,GridFR format - use GridFR as grid position"""
    norm_headers = [h.lower().strip() for h in headers]
    
    time_gap_col = next((i for i, h in enumerate(norm_headers) if 'time/gap' in h), None)
    gridfr_col = next((i for i, h in enumerate(norm_headers) if 'gridfr' in h), None)
    
    result = {}
    
    if time_gap_col is not None and len(row) > time_gap_col:
        result['time'] = row[time_gap_col]
    
    if gridfr_col is not None and len(row) > gridfr_col:
        result['grid'] = row[gridfr_col]
    
    return result


def extract_race_report_links(soup):
    """Extract race report links from the season summary table"""
    # Find Season summary heading
    season_heading = (soup.find("h3", {"id": "Season_summary"}) or
                      soup.find("h3", {"id": "Summary"}) or
                      soup.find("h2", {"id": "Results"}))

    if not season_heading:
        print("No season summary table found")
        return []

    table = season_heading.find_next("table", {"class": "wikitable"})
    if not table:
        print("No season summary table found")
        return []

    race_links = []
    for row in table.find_all("tr")[1:]:
        # Look for Report column (usually last column)
        for cell in row.find_all(["td", "th"]):
            for link in cell.find_all("a"):
                if link.get_text(strip=True).lower() == "report":
                    href = link.get("href")
                    if href and href.startswith("/wiki/"):
                        race_links.append("https://en.wikipedia.org" + href)
                    break

    return race_links


def process_qualifying_data(race_url, round_info):
    """Process qualifying data from a race report page"""
    try:
        response = requests.get(race_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find Qualifying heading
        qualifying_heading = (soup.find("h3", {"id": "Qualifying"}) or
                              soup.find("h3", {"id": "Qualifying_classification"}) or
                              soup.find("h2", {"id": "Qualifying"}))
        if not qualifying_heading:
            print(f"No qualifying section found for {race_url}")
            return None

        # Check if this is Monte Carlo with Group A and Group B
        group_a_head = soup.find("h4", {"id": "Group_A"}) or soup.find("dt", string="Group A")
        group_b_head = soup.find("h4", {"id": "Group_B"}) or soup.find("dt", string="Group B")

        if group_a_head and group_b_head:
            return process_monte_carlo_qualifying(group_a_head, group_b_head, round_info, race_url)
        else:
            return process_single_qualifying_table(qualifying_heading, round_info, race_url)

    except Exception as e:
        print(f"Error processing qualifying data from {race_url}: {str(e)}")
        return None


def process_monte_carlo_qualifying(group_a_head, group_b_head, round_info, race_url):
    """Process Monte Carlo qualifying with Group A and Group B - alternating by time ranking"""
    try:
        # Process Group A
        group_a_table = group_a_head.find_next("table", {"class": "wikitable"})
        group_a_data = []
        if group_a_table:
            group_a_data = extract_quali_table_data(group_a_table)

        # Process Group B
        group_b_table = group_b_head.find_next("table", {"class": "wikitable"})
        group_b_data = []
        if group_b_table:
            group_b_data = extract_quali_table_data(group_b_table)

        if not group_a_data and not group_b_data:
            print(f"No qualifying data found in either group for {race_url}")
            return None

        # Get headers from the first available table
        headers = []
        if group_a_data:
            headers = group_a_data['headers']
        elif group_b_data:
            headers = group_b_data['headers']

        # Process each group separately first to get times
        group_a_processed = []
        group_b_processed = []
        
        if group_a_data:
            group_a_processed = preprocess_qualifying_data(headers, group_a_data['data'])
            # Mark as Group A
            for driver in group_a_processed:
                driver['group'] = 'A'
        
        if group_b_data:
            group_b_processed = preprocess_qualifying_data(headers, group_b_data['data'])
            # Mark as Group B
            for driver in group_b_processed:
                driver['group'] = 'B'

        # Separate regular drivers from special positions
        regular_a = []
        regular_b = []
        special_drivers = []
        
        for driver in group_a_processed:
            pos = driver.get('position', '')
            if pos and any(x in str(pos).upper() for x in ['DSQ', 'EX', 'NC', 'WD']):
                special_drivers.append(driver)
            else:
                regular_a.append(driver)
        
        for driver in group_b_processed:
            pos = driver.get('position', '')
            if pos and any(x in str(pos).upper() for x in ['DSQ', 'EX', 'NC', 'WD']):
                special_drivers.append(driver)
            else:
                regular_b.append(driver)
        
        # Sort each group by qualifying time (fastest first)
        def get_sort_key(driver):
            time_str = driver.get('time')
            if not time_str or time_str in ['—', '-', '']:
                return float('inf')
            
            time_seconds = time_to_seconds(time_str)
            return time_seconds if time_seconds is not None else float('inf')
        
        regular_a.sort(key=get_sort_key)
        regular_b.sort(key=get_sort_key)
        
        # Determine which group has the overall fastest time
        fastest_a_time = get_sort_key(regular_a[0]) if regular_a else float('inf')
        fastest_b_time = get_sort_key(regular_b[0]) if regular_b else float('inf')
        
        # Build final grid by alternating between groups based on their internal ranking
        final_data = []
        a_idx = 0
        b_idx = 0
        position = 1
        
        # Start with the group that has the fastest overall time
        start_with_a = fastest_a_time <= fastest_b_time
        
        while a_idx < len(regular_a) or b_idx < len(regular_b):
            # Determine which group to use for this position
            if position == 1:
                # First position goes to overall fastest
                use_group_a = start_with_a
            else:
                # Alternate starting with the faster group
                # Position 1: faster group, Position 2: slower group, Position 3: faster group, etc.
                use_group_a = (position % 2 == 1) == start_with_a
            
            if use_group_a and a_idx < len(regular_a):
                driver = regular_a[a_idx].copy()
                driver['position'] = str(position)
                final_data.append(driver)
                a_idx += 1
                position += 1
            elif not use_group_a and b_idx < len(regular_b):
                driver = regular_b[b_idx].copy()
                driver['position'] = str(position)
                final_data.append(driver)
                b_idx += 1
                position += 1
            elif a_idx < len(regular_a):
                # If one group is exhausted, continue with the other
                driver = regular_a[a_idx].copy()
                driver['position'] = str(position)
                final_data.append(driver)
                a_idx += 1
                position += 1
            elif b_idx < len(regular_b):
                driver = regular_b[b_idx].copy()
                driver['position'] = str(position)
                final_data.append(driver)
                b_idx += 1
                position += 1
            else:
                break
        
        # Add special positions at the end
        for driver in special_drivers:
            driver_copy = driver.copy()
            # Keep original position for special statuses
            final_data.append(driver_copy)

        return {
            'headers': headers,
            'data': final_data,
            'round_info': round_info,
            'url': race_url,
            'qualifying_type': 'Monte Carlo Groups'
        }

    except Exception as e:
        print(f"Error processing Monte Carlo qualifying from {race_url}: {str(e)}")
        return None


def process_single_qualifying_table(qualifying_heading, round_info, race_url):
    """Process standard single qualifying table"""
    try:
        table = qualifying_heading.find_next("table", {"class": "wikitable"})
        if not table:
            print(f"No qualifying table found for {race_url}")
            return None

        table_data = extract_quali_table_data(table)
        if not table_data:
            return None

        # Preprocess the data
        processed_data = preprocess_qualifying_data(table_data['headers'], table_data['data'])

        return {
            'headers': table_data['headers'],
            'data': processed_data,
            'round_info': round_info,
            'url': race_url,
            'qualifying_type': 'Standard'
        }

    except Exception as e:
        print(f"Error processing single qualifying table: {race_url}: {str(e)}")
        return None


def extract_quali_table_data(table):
    """Extract data from a qualifying table"""
    try:
        all_rows = table.find_all("tr")
        if len(all_rows) < 2:
            return None

        # Get headers
        header_row = all_rows[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

        # Get data rows
        data_rows = []
        for row in all_rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:  # Need at least Pos, No, Driver
                continue

            row_data = []
            for cell in cells:
                # Clean up cell text, remove flag icons and links
                text = cell.get_text(strip=True)
                row_data.append(text)

            if row_data:
                data_rows.append(row_data)

        return {'headers': headers, 'data': data_rows}

    except Exception as e:
        print(f"Error extracting table data: {str(e)}")
        return None


def save_qualifying_to_db(quali_results, year, formula):
    """Save qualifying data to database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('''
            DELETE FROM qualifying 
            WHERE year = %s AND formula = %s
        ''', (year, formula))
        
        for round_num, result in enumerate(quali_results, 1):
            if result is None:
                continue
                
            data = result.get('data', [])
            qualifying_type = result.get('qualifying_type', 'Standard')
            url = result.get('url', '')
            
            for row in data:
                cursor.execute('''
                    INSERT INTO qualifying 
                    (year, formula, round_number, position, car_number, driver, team, time, qualifying_type, url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (year, formula, round_num, row.get('position'), row.get('car_number'), 
                      row.get('driver'), row.get('team'), row.get('time'), qualifying_type, url))
        
        conn.commit()
        print(f"Saved qualifying data for F{formula} {year}")


def scrape_quali(soup, year, num):
    race_links = extract_race_report_links(soup)
    if race_links:
        quali_results = []
        for i, link in enumerate(race_links, 1):
            result = process_qualifying_data(link, f"Round {i}")
            quali_results.append(result)

        save_qualifying_to_db(quali_results, year, num)
        print(f"Saved {len([r for r in quali_results if r])} qualifying sessions to DB")
    else:
        print(f"No race report links found for F{num} {year}")