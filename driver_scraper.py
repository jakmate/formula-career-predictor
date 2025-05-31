import glob
import os
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re


class DriverProfileScraper:
    def __init__(self, profiles_dir="driver_profiles"):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)

        # Academy keywords to search for
        self.academy_keywords = [
            "ferrari driver academy", "red bull junior", "mercedes junior",
            "mclaren young driver", "alpine academy", "williams academy",
            "aston martin junior", "alphatauri junior", "haas development",
            "sauber junior", "renault sport academy"
        ]

    def get_driver_filename(self, driver_name):
        """Create safe filename from driver name."""
        safe_name = re.sub(r'[^\w\s-]', '', driver_name)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        return f"{safe_name.lower()}.json"

    def search_wikipedia_page(self, driver_name):
        """Search for driver's Wikipedia page."""
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"

        # Try different name variations
        name_variations = [
            driver_name,
            driver_name.replace(" ", "_"),
            f"{driver_name}_(racing_driver)",
            f"{driver_name}_(driver)"
        ]

        for variation in name_variations:
            try:
                response = requests.get(f"{search_url}{variation}")
                if response.status_code == 200:
                    data = response.json()
                    if 'extract' in data and len(data['extract']) > 50:
                        return data.get(
                            'content_urls', {}).get(
                            'desktop', {}).get('page')
            except BaseException:
                continue

        return None

    def scrape_driver_profile(self, driver_name):
        """Scrape individual driver profile from Wikipedia."""
        profile_file = os.path.join(
            self.profiles_dir,
            self.get_driver_filename(driver_name)
        )

        # Check if already scraped
        if os.path.exists(profile_file):
            with open(profile_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        print(f"Scraping profile for {driver_name}...")

        # Find Wikipedia page
        wiki_url = self.search_wikipedia_page(driver_name)
        if not wiki_url:
            print(f"No Wikipedia page found for {driver_name}")
            profile = {
                "name": driver_name,
                "dob": None,
                "academy": None,
                "scraped": False
            }
            self.save_profile(profile_file, profile)
            return profile

        try:
            response = requests.get(wiki_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract date of birth
            dob = self.extract_dob(soup)

            # Extract academy information
            academy = self.extract_academy_info(soup)

            profile = {
                "name": driver_name,
                "dob": dob,
                "academy": academy,
                "wiki_url": wiki_url,
                "scraped": True,
                "scraped_date": datetime.now().isoformat()
            }

            self.save_profile(profile_file, profile)
            time.sleep(1)  # Be respectful to Wikipedia
            return profile

        except Exception as e:
            print(f"Error scraping {driver_name}: {e}")
            profile = {
                "name": driver_name,
                "dob": None,
                "academy": None,
                "scraped": False,
                "error": str(e)
            }
            self.save_profile(profile_file, profile)
            return profile

    def extract_dob(self, soup):
        """Extract date of birth from Wikipedia page."""
        # Try multiple selectors for DOB
        selectors = [
            '.bday',  # Standard birthday class
            # Sort values with years
            '[data-sort-value*="19"], [data-sort-value*="20"]',
            'time[datetime]'  # Time elements with datetime
        ]

        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                datetime_attr = element.get('datetime')

                # Try datetime attribute first
                if datetime_attr:
                    try:
                        # Handle various datetime formats
                        if len(datetime_attr) >= 10:  # YYYY-MM-DD format
                            return datetime_attr[:10]
                    except BaseException:
                        pass

                # Try parsing text content
                dob_patterns = [
                    r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
                    # DD Month YYYY
                    r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',  # noqa: 501
                    # Month DD, YYYY
                    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})'  # noqa: 501
                ]

                for pattern in dob_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            if pattern == dob_patterns[0]:  # YYYY-MM-DD
                                return f"""{match.group(1)}
                                -{match.group(2)}-{match.group(3)}"""
                            else:  # Convert month names to numbers
                                month_map = {
                                    'january': '01', 'february': '02',
                                    'march': '03', 'april': '04',
                                    'may': '05', 'june': '06',
                                    'july': '07', 'august': '08',
                                    'september': '09', 'october': '10',
                                    'november': '11', 'december': '12'
                                }
                                if pattern == dob_patterns[1]:  # DD Month YYYY
                                    day = match.group(1).zfill(2)
                                    month = month_map.get(
                                        match.group(2).lower())
                                    year = match.group(3)
                                else:  # Month DD, YYYY
                                    month = month_map.get(
                                        match.group(1).lower())
                                    day = match.group(2).zfill(2)
                                    year = match.group(3)

                                if month:
                                    return f"{year}-{month}-{day}"
                        except BaseException:
                            continue

        # Last resort search in infobox or first paragraph for birth year
        infobox = soup.find('table', class_='infobox')
        if infobox:
            text = infobox.get_text()
            year_match = re.search(r'born.*?(\d{4})', text, re.IGNORECASE)
            if year_match:
                return f"{year_match.group(1)}-01-01"  # Approximate

        return None

    def extract_academy_info(self, soup):
        """Extract specific academy/junior team information."""
        page_text = soup.get_text().lower()

        # Define specific academy names to search for
        specific_academies = [
            "ferrari driver academy",
            "red bull junior team",
            "red bull junior programme",
            "mercedes junior team",
            "mercedes junior programme",
            "mclaren young driver programme",
            "mclaren driver development programme",
            "alpine academy",
            "williams driver academy",
            "williams racing academy",
            "aston martin driver development programme",
            "aston martin junior team",
            "haas development programme",
            "sauber junior team",
            "renault sport academy",
            "lotus f1 junior team",
            "force india development programme"
        ]

        # Search for specific academies first
        for academy in specific_academies:
            if academy in page_text:
                return academy.title()

        # If no specific academy found, return None instead of generic terms
        return None

    def save_profile(self, filename, profile):
        """Save profile to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    def calculate_age(self, dob_str, competition_year):
        """Calculate driver age during competition year."""
        if not dob_str:
            return None

        try:
            # Parse DOB
            if len(dob_str) == 10:  # YYYY-MM-DD
                dob = datetime.strptime(dob_str, '%Y-%m-%d')
            else:
                return None

            # Calculate age at start of season (assume January 1st)
            season_start = datetime(competition_year, 1, 1)
            age = (season_start - dob).days / 365.25
            return round(age, 1)

        except BaseException:
            return None

    def batch_scrape_drivers(self, driver_list):
        """Scrape profiles for list of drivers."""
        profiles = {}
        unique_drivers = list(set(driver_list))

        print(f"Scraping profiles for {len(unique_drivers)} unique drivers...")

        for i, driver in enumerate(unique_drivers, 1):
            print(f"Progress: {i}/{len(unique_drivers)}")
            profiles[driver] = self.scrape_driver_profile(driver)

        return profiles


def clean_driver_name(name):
    """Remove [num] references and other artifacts from driver names."""
    if pd.isna(name) or not isinstance(name, str):
        return None

    # Remove [number] and [letter] references
    cleaned = re.sub(r'\[[^\]]+\]', '', name)

    # Remove other common artifacts
    cleaned = re.sub(r'\d+–\d+', '', cleaned)

    # Strip whitespace and check if valid
    cleaned = cleaned.strip()

    # Skip if empty, only numbers, or clearly not a name
    if not cleaned or cleaned.isdigit() or len(cleaned) < 3:
        return None

    return cleaned


def get_all_drivers_from_data():
    """Extract all driver names from F2 and F3 data files."""
    all_drivers = set()

    # Scan F2 and F3 directories
    for series in ['F2', 'F3']:
        series_dirs = glob.glob(f"{series}/*")

        for year_dir in series_dirs:
            year = os.path.basename(year_dir)
            if not year.isdigit():
                continue

            # Check driver standings files
            standings_file = os.path.join(
                year_dir, f"{series.lower()}_{year}_drivers_standings.csv")
            if os.path.exists(standings_file):
                try:
                    df = pd.read_csv(standings_file)
                    if 'Driver' in df.columns:
                        drivers = df['Driver'].dropna().apply(
                            clean_driver_name).dropna().str.strip().unique()
                        all_drivers.update(drivers)
                except Exception as e:
                    print(f"Error reading {standings_file}: {e}")

            # Check entries files
            entries_file = os.path.join(
                year_dir, f"{series.lower()}_{year}_entries.csv")
            if os.path.exists(entries_file):
                try:
                    df = pd.read_csv(entries_file)
                    # Handle different column names
                    driver_cols = ['Driver', 'Driver name', 'Drivers']
                    for col in driver_cols:
                        if col in df.columns:
                            drivers = df[col].dropna().apply(
                                clean_driver_name
                            ).dropna().str.strip().unique()
                            all_drivers.update(drivers)
                            break
                except Exception as e:
                    print(f"Error reading {entries_file}: {e}")

    return sorted(list(all_drivers))


def main():
    print("Scanning F2/F3 data files for driver names...")
    all_drivers = get_all_drivers_from_data()

    if not all_drivers:
        print("No drivers found. Check your data directory structure.")
        return

    print(f"Found {len(all_drivers)} unique drivers")

    scraper = DriverProfileScraper()

    print("Starting driver profile scraping...")
    profiles = scraper.batch_scrape_drivers(all_drivers)

    scraped_count = sum(
        1 for p in profiles.values() if p.get(
            'scraped', False))
    with_dob = sum(1 for p in profiles.values() if p.get('dob'))
    with_academy = sum(1 for p in profiles.values() if p.get('academy'))

    print("Scraping complete!")
    print(f"Successfully scraped: {scraped_count}/{len(all_drivers)}")
    print(f"Drivers with DOB: {with_dob}")
    print(f"Drivers with academy: {with_academy}")
    print(f"Profiles saved to: {scraper.profiles_dir}/")


if __name__ == "__main__":
    main()
