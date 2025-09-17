import os
import json
import pytz
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from urllib.parse import urljoin

from app.config import CURRENT_YEAR, SCHEDULE_DIR


os.makedirs(SCHEDULE_DIR, exist_ok=True)
TRACK_TIMEZONES = {
    "Sakhir": "Asia/Bahrain",
    "Barcelona": "Europe/Madrid",
    "Imola": "Europe/Rome",
    "Monaco": "Europe/Monaco",
    "Spielberg": "Europe/Vienna",
    "Silverstone": "Europe/London",
    "Budapest": "Europe/Budapest",
    "Spa-Francorchamps": "Europe/Brussels",
    "Zandvoort": "Europe/Amsterdam",
    "Monza": "Europe/Rome",
    "Yas Island": "Asia/Dubai",
    "Jeddah": "Asia/Riyadh",
    "Baku": "Asia/Baku",
    "Melbourne": "Australia/Melbourne",
    "Lusail": "Asia/Doha"
}
TRACK_COUNTRIES = {
    "Sakhir": "Bahrain",
    "Barcelona": "Spain",
    "Imola": "Italy",
    "Monaco": "Monaco",
    "Spielberg": "Austria",
    "Silverstone": "United Kingdom",
    "Budapest": "Hungary",
    "Spa-Francorchamps": "Belgium",
    "Zandvoort": "Netherlands",
    "Monza": "Italy",
    "Yas Island": "United Arab Emirates",
    "Jeddah": "Saudi Arabia",
    "Baku": "Azerbaijan",
    "Melbourne": "Australia",
    "Lusail": "Qatar",
    "Las Vegas": 'United States',
    "Miami": 'United States',
    "Emilia-Romagna": 'Italy',
    "Abu Dhabi": "United Arab Emirates"
}


session = requests.Session()


def get_country_for_location(location_str):
    country = TRACK_COUNTRIES.get(location_str)
    if country:
        return country
    return location_str


def get_timezone_for_location(location_str):
    tz = TRACK_TIMEZONES.get(location_str)
    if tz:
        return tz

    try:
        geolocator = Nominatim(user_agent="f1_schedule_scraper")
        location = geolocator.geocode(location_str, exactly_one=True, timeout=10)
        if location:
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(lat=location.latitude, lng=location.longitude)
            if timezone_str:
                return timezone_str
    except Exception as e:
        print(f"Geocoding error for '{location_str}': {str(e)}")

    # Fallback to UTC if geocoding fails
    return "UTC"


def format_utc_datetime(dt):
    """Format datetime object to ISO string without UTC offset"""
    return dt.replace(tzinfo=None).isoformat() if dt.tzinfo else dt.isoformat()


def is_race_completed_or_ongoing(race):
    """Check if a race is completed or currently ongoing"""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    sessions = race.get('sessions', {})
    if not sessions:
        return False

    first_session = next(iter(sessions.values()))
    start_str = first_session.get('start')
    if not start_str:
        return False

    try:
        if 'T' in start_str:
            session_dt = datetime.fromisoformat(start_str)
            # Convert to naive datetime if it has timezone info
            if session_dt.tzinfo is not None:
                session_dt = session_dt.replace(tzinfo=None)
        else:
            # Date only
            date_val = datetime.strptime(start_str, "%Y-%m-%d").date()
            session_dt = datetime.combine(date_val, datetime.min.time())
    except Exception as e:
        print(e)
        return False

    # Race is completed/ongoing if its weekend has started
    return now >= session_dt


def parse_time_to_datetime(time_str, base_date, day_name=None, location=None):
    """Parse time string and combine with date to create datetime object in UTC"""
    if not time_str:
        return None

    if time_str.upper() == 'TBC':
        result_date = base_date.date()
        if day_name:
            day_mapping = {
                'friday': 4, 'saturday': 5, 'sunday': 6
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
        url = f"https://www.formula1.com/en/racing/{CURRENT_YEAR}.html"
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'lxml')
        response.close()
        del response

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
                    date_obj = datetime.strptime(f"{last_date} {CURRENT_YEAR}", "%d %b %Y")
                else:
                    date_obj = datetime.strptime(f"{date_str} {CURRENT_YEAR}", "%d %b %Y")

                # Get race URL
                race_url = card.get('href')
                sessions = {}

                if race_url:
                    try:
                        full_url = urljoin("https://www.formula1.com", race_url)
                        race_response = session.get(full_url, timeout=10)
                        race_soup = BeautifulSoup(race_response.content, 'lxml')
                        race_response.close()
                        del race_response

                        session_elements = race_soup.select('ul > li[role="listitem"]')

                        if not session_elements:
                            # Fallback try the grid container if the above fails
                            session_elements = race_soup.select('ul.grid > li.relative')

                        session_mapping = {
                            'practice 1': 'fp1',
                            'practice 2': 'fp2',
                            'practice 3': 'fp3',
                            'sprint qualifying': 'sprint_qualifying',
                            'qualifying': 'qualifying',
                            'sprint': 'sprint',
                            'race': 'race'
                        }

                        for session_el in session_elements:
                            # Extract date
                            date_container = session_el.select_one('.min-w-\\[44px\\]')
                            if date_container:
                                day = date_container.select_one('.typography-module_technical-l-bold__AKrZb').text.strip() # noqa: 501
                                month = date_container.select_one('.typography-module_technical-s-regular__6LvKq').text.strip() # noqa: 501
                                session_date = datetime.strptime(f"{day} {month} {CURRENT_YEAR}", "%d %b %Y") # noqa: 501
                            else:
                                # Fallback to the race's main date if session date isn't found
                                session_date = date_obj

                            # Extract session name
                            session_name_el = session_el.select_one('.typography-module_display-m-bold__qgZFB') # noqa: 501
                            session_name = session_name_el.text.strip().lower() if session_name_el else "" # noqa: 501

                            # Extract time
                            time_span = session_el.select_one('.typography-module_technical-s-regular__6LvKq.text-text-5') # noqa: 501
                            time_str = ""
                            if time_span:
                                # Get the raw text and clean it
                                time_text = time_span.get_text(strip=True)
                                # Remove any <time> tags if they are still present as text
                                time_str = re.sub(r'<time[^>]*>|</time>', '', time_text).strip()

                            session_info = None
                            if time_str and '-' in time_str:
                                # Handle time range (e.g., "09:30 - 10:30")
                                start_str, end_str = time_str.split('-', 1)
                                try:
                                    start_time = datetime.strptime(start_str.strip(), "%H:%M").time() # noqa: 501
                                    end_time = datetime.strptime(end_str.strip(), "%H:%M").time()

                                    start_dt = datetime.combine(session_date.date(), start_time)
                                    end_dt = datetime.combine(session_date.date(), end_time)

                                    session_info = {
                                        "start": start_dt.isoformat(),
                                        "end": end_dt.isoformat()
                                    }
                                except ValueError:
                                    # If parsing fails, mark as TBC
                                    session_info = {"start": session_date.isoformat(), "time": "TBC"} # noqa: 501
                            elif time_str:
                                # Handle single time (e.g., Race start time "12:00")
                                try:
                                    start_time = datetime.strptime(time_str.strip(), "%H:%M").time()
                                    start_dt = datetime.combine(session_date.date(), start_time)

                                    # Estimate end time based on session type
                                    if 'race' in session_name:
                                        end_dt = start_dt + timedelta(hours=2)
                                    else:
                                        end_dt = start_dt + timedelta(hours=1)

                                    session_info = {
                                        "start": start_dt.isoformat(),
                                        "end": end_dt.isoformat()
                                    }
                                except ValueError:
                                    session_info = {"start": session_date.isoformat(), "time": "TBC"} # noqa: 501
                            else:
                                # No time found
                                session_info = {"start": session_date.isoformat(), "time": "TBC"}

                            # Map session name to key
                            session_key = None
                            for name_pattern, key in session_mapping.items():
                                if name_pattern in session_name:
                                    session_key = key
                                    break

                            if session_key and session_info:
                                sessions[session_key] = session_info

                        race_soup.decompose()
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
                    "location": get_country_for_location(location),
                    "sessions": sessions
                })
            except Exception as e:
                print(f"Error processing F1 race card: {e}")

        soup.decompose()
        return races
    except Exception as e:
        print(f"Error scraping F1 schedule: {e}")
        return []


def scrape_fia_formula_schedule(series_name):
    """Generic scraper for F2 and F3 schedules"""
    series_config = {
        'f2': {'base_url': 'https://www.fiaformula2.com'},
        'f3': {'base_url': 'https://www.fiaformula3.com'}
    }

    config = series_config.get(series_name)
    if not config:
        raise ValueError(f"Unsupported series: {series_name}")

    try:
        url = f"{config['base_url']}/Calendar"
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'lxml')
        response.close()
        del response

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
                    end_date = date_p.select_one('.end-date').text.strip()
                    month = date_p.select_one('.month').text.strip()
                    date_str = f"{end_date} {month} {CURRENT_YEAR}"

                    race_date = datetime.strptime(date_str, "%d %B %Y")
                else:
                    continue

                # Get race details URL
                race_link = container.select_one('a')
                sessions = {}

                if race_link and race_link.get('href'):
                    try:
                        detail_url = urljoin(config['base_url'], race_link.get('href'))

                        detail_response = session.get(detail_url, timeout=10)
                        detail_soup = BeautifulSoup(detail_response.content, 'lxml')
                        detail_response.close()
                        del detail_response

                        # Look for session schedule
                        session_pins = detail_soup.select('.pin')

                        for pin in session_pins:
                            session_divs = pin.select('div')
                            if (len(session_divs) < 2 or pin.select_one('.position')):
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
                                    location
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

                        detail_soup.decompose()
                    except Exception as e:
                        print(f"Error scraping {series_name.upper()} race details for round {round_num}: {e}") # noqa: 501

                # Add fallback race session
                if 'race' not in sessions:
                    sessions["race"] = {"start": race_date.isoformat()}

                races.append({
                    "round": round_num,
                    "name": location,
                    "location": get_country_for_location(location),
                    "sessions": sessions
                })
            except Exception as e:
                print(f"Error processing {series_name.upper()} race container: {e}")

        soup.decompose()
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

            # Merge schedules - preserve races that are completed or ongoing
            merged_schedule = []
            for new_race in new_schedule:
                round_num = new_race['round']

                # Check if this race is completed or ongoing
                existing_race = existing_races.get(round_num)
                if existing_race and is_race_completed_or_ongoing(existing_race):
                    merged_schedule.append(existing_race)
                else:
                    merged_schedule.append(new_race)

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


if __name__ == "__main__":  # pragma: no cover
    save_schedules()
