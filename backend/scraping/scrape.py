from datetime import datetime
import requests
from bs4 import BeautifulSoup
from scraping.championship_scraper import process_championship
from scraping.entries_scraper import process_entries
from scraping.qualifying_scraper import scrape_quali
from scraping.driver_scraper import scrape_drivers

BASE_URL = "https://en.wikipedia.org/wiki/"


def scrape():
    # F2 and F3/GP3 processing
    for num in [2, 3]:
        for year in range(2010, 2026):
            if (year <= 2016 and num == 2) or (year <= 2018 and num == 3):
                url = f"{BASE_URL}{year}_GP{num}_Series"
            else:
                if num == 3:
                    url = f"{BASE_URL}{year}_FIA_Formula_{num}_Championship"
                elif num == 2:
                    url = f"{BASE_URL}{year}_Formula_{num}_Championship"
                else:
                    print("Error: Invalid formula number")
                    continue

            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                process_entries(soup, year, num)
                process_championship(soup, "Teams'", year, "teams_standings", num)
                process_championship(soup, "Drivers'", year, "drivers_standings", num)
                scrape_quali(soup, year, num)

            except Exception as e:
                print(f"Error processing {year}: {str(e)}")

    # F3 European Championship processing (2012-2018)
    for year in range(2012, 2019):
        url = f"{BASE_URL}{year}_FIA_Formula_3_European_Championship"

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            process_entries(soup, year, 3, "f3_euro")
            process_championship(soup, "Teams'", year, "teams_standings", 3, "f3_euro")
            process_championship(soup, "Drivers'", year, "drivers_standings", 3, "f3_euro")

        except Exception as e:
            print(f"Error processing F3 European {year}: {str(e)}")

    scrape_drivers()


def scrape_current_year():
    current_year = datetime.now().year

    for num in [2, 3]:
        if num == 2:
            url = f"{BASE_URL}{current_year}_Formula_{num}_Championship"
        else:
            url = f"{BASE_URL}{current_year}_FIA_Formula_{num}_Championship"

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            process_entries(soup, current_year, num)
            process_championship(soup, "Teams'", current_year, "teams_standings", num)
            process_championship(soup, "Drivers'", current_year, "drivers_standings", num)
            scrape_quali(soup, current_year, num)

        except Exception as e:
            print(f"Error processing current year F{num}: {str(e)}")

    scrape_drivers()


if __name__ == "__main__":
    scrape()
