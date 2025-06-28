import os
import json
import pytz
import re
import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from urllib.parse import urljoin


SCHEDULE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'schedules', '2025')
os.makedirs(SCHEDULE_DIR, exist_ok=True)

# Initialize geocoding cache
location_timezone_cache = {}


def get_timezone_for_location(location_str):
    if location_str in location_timezone_cache:
        return location_timezone_cache[location_str]

    try:
        geolocator = Nominatim(user_agent="f1_schedule_scraper")
        location = geolocator.geocode(location_str, exactly_one=True, timeout=10)
        if location:
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(lat=location.latitude, lng=location.longitude)
            if timezone_str:
                location_timezone_cache[location_str] = timezone_str
                return timezone_str
    except Exception as e:
        print(f"Geocoding error for '{location_str}': {str(e)}")

    # Fallback to UTC if geocoding fails
    return "UTC"


def format_utc_datetime(dt):
    """Format datetime object to ISO string without UTC offset"""
    return dt.replace(tzinfo=None).isoformat() if dt.tzinfo else dt.isoformat()


def is_race_in_progress(race):
    """Check if a race is currently in progress (during its race weekend)"""
    now = datetime.utcnow()

    # Get all session start times
    session_starts = []
    for session in race['sessions'].values():
        start_str = session.get('start')
        if not start_str:
            continue

        # Handle TBC sessions
        if isinstance(start_str, dict) and start_str.get('time') == 'TBC':
            try:
                date_val = datetime.strptime(start_str['start'], "%Y-%m-%d").date()
                # Use start of day for comparison
                session_starts.append(datetime.combine(date_val, datetime.min.time()))
            except Exception as e:
                continue
        # Handle regular datetime strings
        else:
            try:
                if 'T' in start_str:
                    session_dt = datetime.fromisoformat(start_str)
                else:
                    # Date only
                    date_val = datetime.strptime(start_str, "%Y-%m-%d").date()
                    session_dt = datetime.combine(date_val, datetime.min.time())
                session_starts.append(session_dt)
            except Exception as e:
                continue

    if not session_starts:
        return False

    # Find earliest and latest session times
    earliest_session = min(session_starts)
    latest_session = max(session_starts)

    # Check if current time is within the race weekend
    return earliest_session <= now <= latest_session + timedelta(days=1)


def parse_time_to_datetime(time_str, base_date, day_name=None, location=None):
    """Parse time string and combine with date to create datetime object in UTC"""
    if not time_str:
        return None

    if time_str.upper() == 'TBC':
        result_date = base_date.date()
        if day_name:
            day_mapping = {
                'friday': 4, 'saturday': 5, 'sunday': 6,
                'fri': 4, 'sat': 5, 'sun': 6
            }
            target_weekday = day_mapping.get(day_name.lower())
            if target_weekday is not None:
                base_weekday = base_date.weekday()
                days_diff = target_weekday - base_weekday
                if days_diff > 0:
                    days_diff = days_diff - 7
                result_date = base_date.date() + timedelta(days=days_diff)
        return {"start": result_date.strftime("%Y-%m-%d"), "time": "TBC"}

    try:
        # Handle time ranges
        start_time = end_time = None
        if '-' in time_str:
            start_str, end_str = time_str.split('-', 1)
            start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
            end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
        else:
            start_time = datetime.strptime(time_str.strip(), "%H:%M").time()

        # Start with base date
        result_date = base_date.date()

        # Adjust date based on day name
        if day_name:
            day_mapping = {
                'friday': 4, 'saturday': 5, 'sunday': 6,
                'fri': 4, 'sat': 5, 'sun': 6
            }
            target_weekday = day_mapping.get(day_name.lower())
            if target_weekday is not None:
                base_weekday = base_date.weekday()
                days_diff = target_weekday - base_weekday
                if days_diff > 0:
                    days_diff = days_diff - 7
                result_date = base_date.date() + timedelta(days=days_diff)

        # Create naive datetime objects
        start_dt = datetime.combine(result_date, start_time)
        if end_time:
            end_dt = datetime.combine(result_date, end_time)

        # Convert to UTC if location is provided
        if location:
            tz_str = get_timezone_for_location(location)
            tz = pytz.timezone(tz_str)
            
            # Localize and convert to UTC
            start_dt = tz.localize(start_dt).astimezone(pytz.utc)
            if end_time:
                end_dt = tz.localize(end_dt).astimezone(pytz.utc)

        # Prepare result
        result = {"start": format_utc_datetime(start_dt)}
        if end_time:
            result["end"] = format_utc_datetime(end_dt)

        return result
    except Exception as e:
        print(f"Error parsing time '{time_str}' with day '{day_name}': {e}")
        return None


def scrape_f1_schedule():
    try:
        url = "https://www.formula1.com/en/racing/2025.html"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')

        races = []
        race_cards = soup.select('a.group')

        for card in race_cards:
            try:
                # Extract round number
                round_tag = card.select_one('.typography-module_body-2-xs-bold__M03Ei')
                if not round_tag or "ROUND" not in round_tag.text:
                    continue
                round_num = int(round_tag.text.split()[-1])

                # Extract race name and location
                name_tag = card.select_one('.typography-module_display-xl-bold__Gyl5W')
                location = name_tag.text.strip() if name_tag else "Unknown"

                full_name_tag = card.select_one('.typography-module_body-xs-semibold__Fyfwn')
                race_name = full_name_tag.text.strip() if full_name_tag else "Unknown"

                # Clean race name
                clean_name = re.sub(r'FORMULA 1 |GRAND PRIX|\d{4}', '', race_name).strip()

                # Extract date
                date_span = card.select_one('.typography-module_technical-xs-regular__-W0Gs')
                if not date_span:
                    date_span = card.select_one('.typography-module_technical-m-bold__JDsxP')
                date_str = date_span.text.strip() if date_span else ""

                # Parse date range
                if '-' in date_str:
                    last_date = date_str.split('-')[-1].strip()
                    date_obj = datetime.strptime(f"{last_date} 2025", "%d %b %Y")
                else:
                    date_obj = datetime.strptime(f"{date_str} 2025", "%d %b %Y")

                # Get race URL
                race_url = card.get('href')
                sessions = {}

                if race_url:
                    try:
                        full_url = urljoin("https://www.formula1.com", race_url)
                        race_response = requests.get(full_url, headers={'User-Agent': 'Mozilla/5.0'})
                        race_soup = BeautifulSoup(race_response.content, 'html.parser')

                        # Extract session times from JSON-LD data (UTC times)
                        script_tags = race_soup.find_all('script', type='application/ld+json')
                        events = []
                        for script in script_tags:
                            try:
                                data = json.loads(script.string)
                                if isinstance(data, list):
                                    events.extend(data)
                                else:
                                    events.append(data)
                            except:
                                continue

                        session_mapping = {
                            'Practice 1': 'fp1',
                            'Practice 2': 'fp2',
                            'Practice 3': 'fp3',
                            'Qualifying': 'qualifying',
                            'Sprint Qualifying': 'sprint_qualifying',
                            'Sprint Shootout': 'sprint_qualifying',
                            'Sprint': 'sprint',
                            'Race': 'race'
                        }

                        for event in events:
                            if isinstance(event, dict) and event.get('@type') == 'SportsEvent':
                                name = event.get('name', '')
                                for pattern, key in session_mapping.items():
                                    if pattern in name:
                                        start = event.get('startDate')
                                        end = event.get('endDate')
                                        if start:
                                            # Handle potential formatting issues
                                            if ' ' in start:
                                                start = start.replace(' ', 'T')
                                            if end and ' ' in end:
                                                end = end.replace(' ', 'T')
                                            
                                            session_info = {"start": start}
                                            if end:
                                                session_info["end"] = end
                                            sessions[key] = session_info
                                        break

                        # Fallback to old scraping method if sessions missing
                        if not sessions:
                            session_container = race_soup.select_one('.flex.flex-col.px-px-8.lg\\:px-px-16.py-px-8.lg\\:py-px-16.bg-surface-neutral-1.rounded-m')
                            if session_container:
                                session_elements = session_container.select('ul.contents > li')
                            else:
                                session_elements = race_soup.select('.schedule-item, .session-item, .event-schedule-item')

                            session_mapping = {
                                'practice 1': 'fp1',
                                'practice 2': 'fp2',
                                'practice 3': 'fp3',
                                'qualifying': 'qualifying',
                                'sprint qualifying': 'sprint_qualifying',
                                'sprint shootout': 'sprint_qualifying',
                                'sprint': 'sprint',
                                'race': 'race'
                            }

                            for session in session_elements:
                                date_container = session.select_one('.min-w-\\[44px\\]')
                                if date_container:
                                    day = date_container.select_one('.typography-module_technical-l-bold__AKrZb').text.strip()
                                    month = date_container.select_one('.typography-module_technical-s-regular__6LvKq').text.strip()
                                    session_date = datetime.strptime(f"{day} {month} 2025", "%d %b %Y")
                                else:
                                    session_date = date_obj

                                session_name = session.select_one('.typography-module_display-m-bold__qgZFB')
                                session_name = session_name.text.strip().lower() if session_name else ""

                                time_span = session.select_one('.typography-module_technical-s-regular__6LvKq.text-text-5')
                                if time_span:
                                    time_str = time_span.get_text(strip=True)
                                    time_str = re.sub(r'<time>|</time>', '', time_str)
                                else:
                                    time_str = ""

                                session_info = None
                                if time_str:
                                    if '-' in time_str:
                                        start_str, end_str = time_str.split('-', 1)
                                        start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
                                        end_time = datetime.strptime(end_str.strip(), "%H:%M").time()

                                        start_dt = datetime.combine(session_date.date(), start_time)
                                        end_dt = datetime.combine(session_date.date(), end_time)

                                        session_info = {
                                            "start": start_dt.isoformat(),
                                            "end": end_dt.isoformat()
                                        }
                                    else:
                                        start_time = datetime.strptime(time_str.strip(), "%H:%M").time()
                                        start_dt = datetime.combine(session_date.date(), start_time)
                                        if 'race' in session_name:
                                            end_dt = start_dt + timedelta(hours=2)
                                            session_info = {
                                                "start": start_dt.isoformat(),
                                                "end": end_dt.isoformat()
                                            }
                                        else:
                                            session_info = {"start": start_dt.isoformat()}
                                else:
                                    session_info = {"start": session_date.isoformat(), "time": "TBC"}

                                session_key = None
                                for name_pattern, key in session_mapping.items():
                                    if name_pattern in session_name:
                                        session_key = key
                                        break

                                if session_key and session_info:
                                    sessions[session_key] = session_info

                        time.sleep(0.5)
                    except Exception as e:
                        print(f"Error scraping F1 race details for round {round_num}: {e}")

                # Add fallback sessions if missing
                if 'race' not in sessions:
                    sessions["race"] = {"start": date_obj.isoformat(), "time": "TBC"}
                if 'qualifying' not in sessions and 'sprint_qualifying' not in sessions:
                    sessions["qualifying"] = {"start": date_obj.isoformat(), "time": "TBC"}

                races.append({
                    "round": round_num,
                    "name": clean_name,
                    "location": location,
                    "sessions": sessions
                })
            except Exception as e:
                print(f"Error processing F1 race card: {e}")

        return races
    except Exception as e:
        print(f"Error scraping F1 schedule: {e}")
        return []


def scrape_fia_formula_schedule(series_name):
    """Generic scraper for F2 and F3 schedules"""
    series_config = {
        'f2': {
            'url': 'https://www.fiaformula2.com/Calendar',
            'base_url': 'https://www.fiaformula2.com'
        },
        'f3': {
            'url': 'https://www.fiaformula3.com/Calendar',
            'base_url': 'https://www.fiaformula3.com'
        }
    }

    config = series_config.get(series_name)
    if not config:
        raise ValueError(f"Unsupported series: {series_name}")

    try:
        response = requests.get(config['url'], headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')

        races = []
        race_containers = soup.select('.col-12.col-sm-6.col-lg-4.col-xl-3')

        for container in race_containers:
            try:
                # Extract round number
                round_tag = container.select_one('.h6')
                if not round_tag or "Round" not in round_tag.text:
                    continue
                round_num = int(round_tag.text.split()[-1])

                # Extract location
                location_span = container.select_one('.event-place .ellipsis')
                location = location_span.text.strip() if location_span else "Unknown"

                # Extract date components
                date_p = container.select_one('p.date')
                if date_p:
                    start_date = date_p.select_one('.start-date').text.strip()
                    end_date = date_p.select_one('.end-date').text.strip()
                    month = date_p.select_one('.month').text.strip()

                    if int(start_date) > int(end_date):
                        date_str = f"{end_date} {month} 2025"
                    else:
                        date_str = f"{end_date} {month} 2025"

                    race_date = datetime.strptime(date_str, "%d %B %Y")
                else:
                    continue

                # Get race details URL
                race_link = container.select_one('a')
                sessions = {}

                if race_link and race_link.get('href'):
                    try:
                        if series_name == 'f3' and 'raceid=' in race_link.get('href'):
                            race_id = race_link.get('href').split('raceid=')[-1]
                            detail_url = f"{config['base_url']}/Results?raceid={race_id}"
                        else:
                            detail_url = urljoin(config['base_url'], race_link.get('href'))

                        detail_response = requests.get(detail_url, headers={'User-Agent': 'Mozilla/5.0'})
                        detail_soup = BeautifulSoup(detail_response.content, 'html.parser')

                        # Look for session schedule
                        session_pins = detail_soup.select('.pin')

                        for pin in session_pins:
                            session_divs = pin.select('div')
                            if (len(session_divs) < 2 or 'Summary' in pin.text or
                                'Standings' in pin.text or 'hint' in pin.get('class', []) or
                                pin.select_one('.position')):
                                continue

                            session_name = session_divs[0].text.strip().lower()
                            driver_span = pin.select_one('.driver-name')
                            if driver_span:
                                fallback_time = {"start": race_date.isoformat()}
                                if 'practice' in session_name:
                                    sessions['practice'] = fallback_time
                                elif 'qualifying' in session_name:
                                    sessions['qualifying'] = fallback_time
                                elif 'sprint' in session_name:
                                    sessions['sprint'] = fallback_time
                                elif 'feature' in session_name:
                                    sessions['race'] = fallback_time
                                continue

                            if len(session_divs) >= 3:
                                day_name = session_divs[1].text.strip().lower()
                                time_span = pin.select_one('.highlight')
                                time_str = time_span.text.strip() if time_span else ""

                                # Pass location to parser
                                session_dt = parse_time_to_datetime(
                                    time_str, 
                                    race_date, 
                                    day_name, 
                                    location  # Pass location for timezone conversion
                                )

                                if session_dt:
                                    if 'practice' in session_name:
                                        sessions['practice'] = session_dt
                                    elif 'qualifying' in session_name:
                                        sessions['qualifying'] = session_dt
                                    elif 'sprint' in session_name:
                                        sessions['sprint'] = session_dt
                                    elif 'feature' in session_name:
                                        sessions['race'] = session_dt

                        time.sleep(0.5)
                    except Exception as e:
                        print(f"Error scraping {series_name.upper()} race details for round {round_num}: {e}")

                # Add fallback race session
                if 'race' not in sessions:
                    sessions["race"] = {"start": race_date.isoformat()}

                races.append({
                    "round": round_num,
                    "name": location,
                    "location": location,
                    "sessions": sessions
                })
            except Exception as e:
                print(f"Error processing {series_name.upper()} race container: {e}")

        return races
    except Exception as e:
        print(f"Error scraping {series_name.upper()} schedule: {e}")
        return []


def save_schedules():
    series_scrapers = {
        'f1': scrape_f1_schedule,
        'f2': lambda: scrape_fia_formula_schedule('f2'),
        'f3': lambda: scrape_fia_formula_schedule('f3')
    }

    for name, scraper in series_scrapers.items():
        try:
            file_path = os.path.join(SCHEDULE_DIR, f"{name}.json")
            existing_schedule = []
            
            # Load existing schedule if available
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    try:
                        existing_schedule = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse existing {name} schedule")
                        existing_schedule = []
            
            # Scrape new schedule
            new_schedule = scraper()
            if not new_schedule:
                print(f"Warning: No races scraped for {name}")
                continue
            
            # Create a map of existing races by round number
            existing_races = {race['round']: race for race in existing_schedule}
            
            # Merge schedules: preserve races that are in progress or completed
            merged_schedule = []
            for new_race in new_schedule:
                round_num = new_race['round']
                
                # Check if this race is in progress
                existing_race = existing_races.get(round_num)
                if existing_race and is_race_in_progress(existing_race):
                    # Preserve existing data for in-progress races
                    merged_schedule.append(existing_race)
                    print(f"Preserving in-progress race: {name} round {round_num}")
                else:
                    # Use new data for upcoming races
                    merged_schedule.append(new_race)
            
            # Add any missing races from existing schedule (shouldn't happen, but just in case)
            existing_rounds = {race['round'] for race in merged_schedule}
            for race in existing_schedule:
                if race['round'] not in existing_rounds:
                    merged_schedule.append(race)
            
            # Sort by round number
            merged_schedule.sort(key=lambda x: x['round'])
            
            # Save only if there are changes
            if merged_schedule != existing_schedule:
                with open(file_path, "w") as f:
                    json.dump(merged_schedule, f, indent=2)
                print(f"Updated {name.upper()} schedule: {len(merged_schedule)} races")
            else:
                print(f"No changes detected for {name.upper()} schedule")
                
        except Exception as e:
            print(f"Error saving {name} schedule: {e}")


if __name__ == "__main__":
    save_schedules()