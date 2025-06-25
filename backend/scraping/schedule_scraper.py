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
    if not time_str or time_str.upper() == 'TBC':
        return None

    try:
        # Handle time ranges (take start time)
        if '-' in time_str:
            time_str = time_str.split('-')[0]

        # Parse time
        time_obj = datetime.strptime(time_str.strip(), "%H:%M").time()

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
        result = datetime.combine(result_date, time_obj)
        return result.isoformat()
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
                sessions = {"gp": date_obj.isoformat()}

                if race_url:
                    try:
                        full_url = urljoin("https://www.formula1.com", race_url)
                        race_response = requests.get(full_url,
                                                     headers={'User-Agent': 'Mozilla/5.0'})
                        race_soup = BeautifulSoup(race_response.content, 'html.parser')

                        # Look for session schedule (F1 has various selectors)
                        session_elements = race_soup.select('.schedule-item, .session-item, .event-schedule-item')  # noqa: 501

                        for session in session_elements:
                            session_name = session.select_one('.session-name, .event-name')
                            session_time = session.select_one('.session-time, .event-time')
                            session_day = session.select_one('.session-day, .event-day')

                            if session_name and session_time:
                                name = session_name.text.strip().lower()
                                time_str = session_time.text.strip()
                                day_str = session_day.text.strip() if session_day else None

                                session_dt = parse_time_to_datetime(time_str, date_obj, day_str)
                                if session_dt:
                                    if 'practice' in name or 'fp' in name:
                                        sessions['practice'] = session_dt
                                    elif 'qualifying' in name or 'quali' in name:
                                        sessions['qualifying'] = session_dt
                                    elif 'sprint' in name:
                                        sessions['sprint'] = session_dt

                        time.sleep(0.5)
                    except Exception as e:
                        print(f"Error scraping F1 race details for round {round_num}: {e}")

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


def scrape_f2_schedule():
    try:
        url = "https://www.fiaformula2.com/Calendar"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
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
                sessions = {"race": race_date.isoformat()}

                if race_link and race_link.get('href'):
                    try:
                        detail_url = urljoin("https://www.fiaformula2.com", race_link.get('href'))
                        detail_response = requests.get(detail_url,
                                                       headers={'User-Agent': 'Mozilla/5.0'})
                        detail_soup = BeautifulSoup(detail_response.content, 'html.parser')

                        # Look for session schedule in F2 format
                        session_pins = detail_soup.select('.pin')

                        for pin in session_pins:
                            session_divs = pin.select('div')

                            # Skip non-session pins (summaries, standings, hints)
                            if (len(session_divs) < 2 or 'Summary' in pin.text or 'Standings' in pin.text or  # noqa: 501
                                'hint' in pin.get('class', []) or pin.select_one('.position')):  # noqa: 501
                                continue

                            session_name = session_divs[0].text.strip().lower()

                            # Case 1: Completed races (driver name + time in highlight)
                            driver_span = pin.select_one('.driver-name')
                            if driver_span:
                                # For completed sessions, create entries without specific times
                                if 'practice' in session_name:
                                    sessions['practice'] = race_date.isoformat()
                                elif 'qualifying' in session_name:
                                    sessions['qualifying'] = race_date.isoformat()
                                elif 'sprint' in session_name:
                                    sessions['sprint'] = race_date.isoformat()
                                elif 'feature' in session_name:
                                    sessions['race'] = race_date.isoformat()
                                continue

                            # Case 2: Future races with times or TBC
                            if len(session_divs) >= 3:
                                day_name = session_divs[1].text.strip().lower()
                                time_span = pin.select_one('.highlight')

                                if time_span:
                                    time_str = time_span.text.strip()
                                    session_dt = parse_time_to_datetime(time_str,
                                                                        race_date,
                                                                        day_name)

                                    # For TBC times, use race date as fallback
                                    if not session_dt and time_str.upper() != 'TBC':
                                        session_dt = race_date.isoformat()
                                    elif not session_dt:
                                        session_dt = race_date.isoformat()  # TBC case

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
                        print(f"Error scraping F2 race details for round {round_num}: {e}")

                races.append({
                    "round": round_num,
                    "name": location,
                    "location": location,
                    "sessions": sessions
                })
            except Exception as e:
                print(f"Error processing F2 race container: {e}")

        return races
    except Exception as e:
        print(f"Error scraping F2 schedule: {e}")
        return []


def scrape_f3_schedule():
    try:
        url = "https://www.fiaformula3.com/Calendar"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
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

                # Get race details URL from the Results link
                race_link = container.select_one('a[href*="/Results"]')
                sessions = {"race": race_date.isoformat()}

                if race_link and race_link.get('href'):
                    try:
                        # Convert Results URL to race details URL
                        results_url = race_link.get('href')
                        if 'raceid=' in results_url:
                            race_id = results_url.split('raceid=')[-1]
                        else:
                            race_id = None

                        if race_id:
                            detail_url = f"https://www.fiaformula3.com/Results?raceid={race_id}"
                            detail_response = requests.get(detail_url,
                                                           headers={'User-Agent': 'Mozilla/5.0'})
                            detail_soup = BeautifulSoup(detail_response.content, 'html.parser')

                            # Look for session schedule in the circuit content
                            session_pins = detail_soup.select('.pin')

                            for pin in session_pins:
                                session_divs = pin.select('div')

                                # Skip non-session pins (summaries, standings, hints)
                                if (len(session_divs) < 2 or 'Summary' in pin.text or 'Standings' in pin.text or  # noqa: 501
                                    'hint' in pin.get('class', []) or pin.select_one('.position')):  # noqa: 501
                                    continue

                                session_name = session_divs[0].text.strip().lower()

                                # Case 1: Completed races (driver name + time in highlight)
                                driver_span = pin.select_one('.driver-name')
                                if driver_span:
                                    # For completed sessions, create entries without specific times
                                    if 'practice' in session_name:
                                        sessions['practice'] = race_date.isoformat()
                                    elif 'qualifying' in session_name:
                                        sessions['qualifying'] = race_date.isoformat()
                                    elif 'sprint' in session_name:
                                        sessions['sprint'] = race_date.isoformat()
                                    elif 'feature' in session_name:
                                        sessions['race'] = race_date.isoformat()
                                    continue

                                # Case 2: Future races with times or TBC
                                if len(session_divs) >= 3:
                                    day_name = session_divs[1].text.strip().lower()
                                    time_span = pin.select_one('.highlight')

                                    if time_span:
                                        time_str = time_span.text.strip()
                                        session_dt = parse_time_to_datetime(time_str,
                                                                            race_date,
                                                                            day_name)

                                        # For TBC times, use race date as fallback
                                        if not session_dt and time_str.upper() != 'TBC':
                                            session_dt = race_date.isoformat()
                                        elif not session_dt:
                                            session_dt = race_date.isoformat()  # TBC case

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
                        print(f"Error scraping F3 race details for round {round_num}: {e}")

                races.append({
                    "round": round_num,
                    "name": location,
                    "location": location,
                    "sessions": sessions
                })
            except Exception as e:
                print(f"Error processing F3 race container: {e}")

        return races
    except Exception as e:
        print(f"Error scraping F3 schedule: {e}")
        return []


def save_schedules():
    series_scrapers = {
        # 'f1': scrape_f1_schedule,
        'f2': scrape_f2_schedule,
        'f3': scrape_f3_schedule
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
