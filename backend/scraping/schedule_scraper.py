import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import time
from urllib.parse import urljoin

SCHEDULE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'schedules', '2025')
os.makedirs(SCHEDULE_DIR, exist_ok=True)


def parse_time_to_datetime(time_str, base_date, day_name=None):
    """Parse time string and combine with date to create datetime object"""
    if not time_str:
        return None

    if time_str.upper() == 'TBC':
        # For TBC times, return the date with TBC indicator
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

        # Start with the base date (usually the race weekend end date)
        result_date = base_date.date()

        # Adjust date based on day name if provided
        if day_name:
            day_mapping = {
                'friday': 4, 'saturday': 5, 'sunday': 6,
                'fri': 4, 'sat': 5, 'sun': 6
            }
            target_weekday = day_mapping.get(day_name.lower())
            if target_weekday is not None:
                # Calculate the date for the specific day of the race weekend
                base_weekday = base_date.weekday()  # Sunday = 6
                days_diff = target_weekday - base_weekday

                # If the target day is before the base date in the same week, go back
                if days_diff > 0:
                    days_diff = days_diff - 7

                result_date = base_date.date() + timedelta(days=days_diff)

        # Combine date and time
        start_dt = datetime.combine(result_date, start_time)
        result = {"start": start_dt.isoformat()}

        if end_time:
            end_dt = datetime.combine(result_date, end_time)
            result["end"] = end_dt.isoformat()

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

                # Parse date range (take the last date as race day)
                if '-' in date_str:
                    last_date = date_str.split('-')[-1].strip()
                    date_obj = datetime.strptime(f"{last_date} 2025", "%d %b %Y")
                else:
                    date_obj = datetime.strptime(f"{date_str} 2025", "%d %b %Y")

                # Get race URL for detailed session info
                race_url = card.get('href')
                sessions = {}

                if race_url:
                    try:
                        full_url = urljoin("https://www.formula1.com", race_url)
                        race_response = requests.get(full_url,
                                                     headers={'User-Agent': 'Mozilla/5.0'})
                        race_soup = BeautifulSoup(race_response.content, 'html.parser')

                        # Look for session schedule in the updated HTML structure
                        session_container = race_soup.select_one('.flex.flex-col.px-px-8.lg\\:px-px-16.py-px-8.lg\\:py-px-16.bg-surface-neutral-1.rounded-m')  # noqa: 501
                        if session_container:
                            session_elements = session_container.select('ul.contents > li')
                        else:
                            # Fallback to old selectors if new structure not found
                            session_elements = race_soup.select('.schedule-item, .session-item, .event-schedule-item')  # noqa: 501

                        # Create session mapping
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
                            # Extract session date components
                            date_container = session.select_one('.min-w-\\[44px\\]')
                            if date_container:
                                day = date_container.select_one('.typography-module_technical-l-bold__AKrZb').text.strip()  # noqa: 501
                                month = date_container.select_one('.typography-module_technical-s-regular__6LvKq').text.strip()  # noqa: 501
                                session_date = datetime.strptime(f"{day} {month} 2025", "%d %b %Y")
                            else:
                                session_date = date_obj  # Fallback to race date

                            # Extract session name
                            session_name = session.select_one('.typography-module_display-m-bold__qgZFB')  # noqa: 501
                            session_name = session_name.text.strip().lower() if session_name else ""

                            # Extract session time
                            time_span = session.select_one('.typography-module_technical-s-regular__6LvKq.text-text-5')  # noqa: 501
                            if time_span:
                                time_str = time_span.get_text(strip=True)
                                # Remove <time> tags if present
                                time_str = re.sub(r'<time>|</time>', '', time_str)
                            else:
                                time_str = ""

                            # Handle time ranges and single times
                            session_info = None
                            if time_str:
                                if '-' in time_str:
                                    start_str, end_str = time_str.split('-', 1)
                                    start_time = datetime.strptime(start_str.strip(), "%H:%M").time()  # noqa: 501
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
                                        # For races, set a default duration of 2 hours
                                        end_dt = start_dt + timedelta(hours=2)
                                        session_info = {
                                            "start": start_dt.isoformat(),
                                            "end": end_dt.isoformat()
                                        }
                                    else:
                                        session_info = {"start": start_dt.isoformat()}
                            else:
                                session_info = {"start": session_date.isoformat(), "time": "TBC"}

                            # Map session names to our keys
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

                # Add race session with fallback
                if 'race' not in sessions:
                    sessions["race"] = {"start": date_obj.isoformat(), "time": "TBC"}

                # Add qualifying with fallback to race date if missing
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
                        # Handle different URL construction methods
                        if series_name == 'f3' and 'raceid=' in race_link.get('href'):
                            race_id = race_link.get('href').split('raceid=')[-1]
                            detail_url = f"{config['base_url']}/Results?raceid={race_id}"
                        else:
                            detail_url = urljoin(config['base_url'], race_link.get('href'))

                        detail_response = requests.get(detail_url,
                                                       headers={'User-Agent': 'Mozilla/5.0'})
                        detail_soup = BeautifulSoup(detail_response.content, 'html.parser')

                        # Look for session schedule
                        session_pins = detail_soup.select('.pin')

                        for pin in session_pins:
                            session_divs = pin.select('div')

                            # Skip non-session pins
                            if (len(session_divs) < 2 or 'Summary' in pin.text or 'Standings' in pin.text or  # noqa: 501
                                'hint' in pin.get('class', []) or pin.select_one('.position')):  # noqa: 501
                                continue

                            session_name = session_divs[0].text.strip().lower()

                            # Case 1: Completed races (driver name + time in highlight)
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

                            # Case 2: Future races with times or TBC
                            if len(session_divs) >= 3:
                                day_name = session_divs[1].text.strip().lower()
                                time_span = pin.select_one('.highlight')

                                if time_span:
                                    time_str = time_span.text.strip()
                                    session_dt = parse_time_to_datetime(time_str, race_date, day_name)  # noqa: 501

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
                        print(f"Error scraping {series_name.upper()} race details for round {round_num}: {e}")  # noqa: 501

                # Only add fallback race session if no feature race was found
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
            schedule = scraper()
            file_path = os.path.join(SCHEDULE_DIR, f"{name}.json")

            with open(file_path, "w") as f:
                json.dump(schedule, f, indent=2)
            print(f"Saved {len(schedule)} races for {name.upper()} to {file_path}")
        except Exception as e:
            print(f"Error saving {name} schedule: {e}")


if __name__ == "__main__":
    save_schedules()
