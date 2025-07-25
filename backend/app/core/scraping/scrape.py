import gc
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from app.core.scraping.schedule_scraper import save_schedules
from app.core.scraping.championship_scraper import process_championship
from app.core.scraping.entries_scraper import process_entries
from app.core.scraping.qualifying_scraper import scrape_quali
from app.core.scraping.driver_scraper import scrape_drivers

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
    session = requests.Session()
    try:
        # F1, F2/GP2 and F3/GP3 processing
        for num in [1, 2, 3]:
            for year in range(2010, 2026):
                url = map_url(num, year)

                try:
                    response = session.get(url, timeout=10)
                    response.raise_for_status()
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
                    print(f"Error processing {year}: {str(e)}")

        # F3 European Championship processing (2012-2018)
        for year in range(2012, 2019):
            url = f"{BASE_URL}{year}_FIA_Formula_3_European_Championship"

            try:
                response = session.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "lxml")
                response.close()
                del response

                process_entries(soup, year, 3, "f3_euro")
                process_championship(soup, "Teams'", year, "teams_standings", 3, "f3_euro")
                process_championship(soup, "Drivers'", year, "drivers_standings", 3, "f3_euro")

                soup.decompose()
                gc.collect()

            except Exception as e:
                print(f"Error processing F3 European {year}: {str(e)}")
    finally:
        session.close()

    scrape_drivers()
    save_schedules()


def scrape_current_year():
    current_year = datetime.now().year
    session = requests.Session()

    try:
        for num in [1, 2, 3]:
            if num == 1:
                url = f"{BASE_URL}{current_year}_Formula_One_World_Championship"
            elif num == 2:
                url = f"{BASE_URL}{current_year}_Formula_{num}_Championship"
            else:
                url = f"{BASE_URL}{current_year}_FIA_Formula_{num}_Championship"

            try:
                response = session.get(url, timeout=10)
                response.raise_for_status()
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


if __name__ == "__main__":
    scrape()
