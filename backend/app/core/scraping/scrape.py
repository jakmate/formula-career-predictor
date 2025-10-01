import gc
from bs4 import BeautifulSoup
from datetime import datetime
from app.core.scraping.schedule_scraper import save_schedules
from app.core.scraping.championship_scraper import process_championship
from app.core.scraping.entries_scraper import process_entries
from app.core.scraping.qualifying_scraper import scrape_quali
from app.core.scraping.driver_scraper import scrape_drivers
from app.core.scraping.scraping_utils import create_session, safe_request

BASE_URL = "https://en.wikipedia.org/wiki/"


def map_url(num, year):
    if num == 1:
        return f"{BASE_URL}{year}_Formula_One_World_Championship"
    elif (year <= 2016 and num == 2) or (year <= 2018 and num == 3):
        return f"{BASE_URL}{year}_GP{num}_Series"
    elif num == 3:
        return f"{BASE_URL}{year}_FIA_Formula_{num}_Championship"
    elif num == 2:
        return f"{BASE_URL}{year}_Formula_{num}_Championship"
    return None


def scrape():
    session = create_session()

    try:
        # F1, F2/GP2 and F3/GP3 processing
        for num in [1, 2, 3]:
            for year in range(2010, 2026):
                url = map_url(num, year)
                print(f"Processing F{num} {year}...")

                response = safe_request(session, url)
                if response is None:
                    print(f"Skipping F{num} {year} due to request failure")
                    continue

                try:
                    soup = BeautifulSoup(response.text, "lxml")
                    response.close()
                    del response

                    process_entries(soup, year, num)
                    process_championship(soup, "Teams'", year, "teams_standings", num)
                    process_championship(soup, "Drivers'", year, "drivers_standings", num)
                    scrape_quali(soup, year, num)

                    soup.decompose()
                    gc.collect()

                except Exception as e:
                    print(f"Error processing data for F{num} {year}: {str(e)}")
    finally:
        session.close()

    scrape_drivers()
    save_schedules()


def scrape_current_year():
    current_year = datetime.now().year
    session = create_session()

    try:
        for num in [1, 2, 3]:
            if num == 1:
                url = f"{BASE_URL}{current_year}_Formula_One_World_Championship"
            elif num == 2:
                url = f"{BASE_URL}{current_year}_Formula_{num}_Championship"
            else:
                url = f"{BASE_URL}{current_year}_FIA_Formula_{num}_Championship"

            print(f"Processing current year F{num}...")
            response = safe_request(session, url)
            if response is None:
                print(f"Skipping current year F{num} due to request failure")
                continue

            try:
                soup = BeautifulSoup(response.text, "lxml")
                response.close()
                del response

                process_entries(soup, current_year, num)
                process_championship(soup, "Teams'", current_year, "teams_standings", num)
                process_championship(soup, "Drivers'", current_year, "drivers_standings", num)
                scrape_quali(soup, current_year, num)

                soup.decompose()
                gc.collect()

            except Exception as e:
                print(f"Error processing current year F{num}: {str(e)}")

    finally:
        session.close()

    scrape_drivers()
    save_schedules()


if __name__ == "__main__":  # pragma: no cover
    scrape()
