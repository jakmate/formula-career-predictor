import glob
import json
import os
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime


class DriverProfileScraper:
    def __init__(self, profiles_dir="data/driver_profiles"):
        self.profiles_dir = profiles_dir
        self.session = requests.Session()
        os.makedirs(profiles_dir, exist_ok=True)

        # Known driver aliases and redirects
        self.driver_aliases = {
            "Peter Li": "Li Zhicong",
            "Hongwei Cao": "Martin Cao",
            "Michael Lewis": "Michael James Lewis",
            "Richard Goddard": "Spike Goddard",
            "Edward Jones": "Ed Jones",
            "Nick McBride": None,
            "Bang Hongwei": None,
            "Sam MacLeod": None,
            "Matthew Rao": None,
            "Pedro Pablo Calbimonte": None,
            "Nicolas Pohler": None,
            "Fahmi Ilyas": None,
            "Andrea Roda": None,
            "Alexander Toril": None,
            "Geoff Uhrhane": None,
            "Ben Hingeley": None,
            "Patrick Reiterer": None,
            "Denis Nagulin": None,
        }

    def get_driver_filename(self, driver_name):
        """Create safe filename from driver name."""
        safe_name = re.sub(r'[^\w\s-]', '', driver_name)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        return f"{safe_name.lower()}.json"

    def is_racing_driver_page(self, soup):
        """Check if the Wikipedia page is about a racing driver."""
        # Racing-related keywords that indicate this is a driver page
        racing_keywords = [
            'formula', 'racing driver', 'motorsport', 'grand prix',
            'championship', 'circuit', 'f1', 'f2', 'f3', 'indycar',
            'nascar', 'lemans', 'endurance racing', 'karting',
            'single-seater', 'open-wheel', 'touring car', 'gp2', 'gp3'
        ]

        # Check if any racing keywords appear in the first few paragraphs
        first_paragraphs = ' '.join([p.get_text().lower()
                                    for p in soup.find_all('p')[:3]])

        return any(keyword in first_paragraphs for keyword in racing_keywords)

    def search_wikipedia_page(self, driver_name):
        """Search driver's Wikipedia page using Wikipedia search API."""
        # Check for known aliases first
        if driver_name in self.driver_aliases:
            alias = self.driver_aliases[driver_name]
            if alias is None:  # Explicitly marked as invalid
                return None
            driver_name = alias

        # First try the OpenSearch API for fuzzy matching
        opensearch_url = "https://en.wikipedia.org/w/api.php"

        # Try OpenSearch API first
        try:
            params = {
                'action': 'opensearch',
                'search': driver_name,
                'limit': 5,
                'format': 'json',
                'redirects': 'resolve',
                'timeout': 10
            }

            response = self.session.get(opensearch_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if len(data) >= 4 and data[3]:  # URLs are in the 4th element
                    urls = data[3]

                    # Check each result to find racing driver pages
                    for i, url in enumerate(urls):
                        try:
                            page_response = self.session.get(url)
                            if page_response.status_code == 200:
                                soup = BeautifulSoup(page_response.text, 'lxml')
                                if self.is_racing_driver_page(soup):
                                    return url
                                del soup
                        except Exception as e:
                            print(f"Error checking search result {i}: {e}")
                            continue

                    # If no racing driver found, return first result as fallback
                    if urls:
                        return urls[0]

        except Exception as e:
            print(f"OpenSearch API error for {driver_name}: {e}")

        return None

    def extract_nationality(self, soup):
        """Extract driver nationality from infobox, otherwise from short description."""
        # Infobox extraction
        infobox = soup.find('table', class_='infobox')
        if infobox:
            for tr in infobox.find_all('tr'):
                th = tr.find('th')
                if th and 'nationality' in th.get_text(strip=True).lower():
                    td = tr.find('td')
                    if td:
                        # Use plain text in that cell
                        text = td.get_text(" ", strip=True)
                        # remove reference marks
                        text = re.sub(r'\[\d+\]', '', text)
                        # strip off any italic parentheticals
                        text = re.sub(r'\s*via .+$', '', text, flags=re.IGNORECASE)
                        # Normalize separators and casing
                        return self.normalize_nationality_text(text)

        # Short description extraction
        shortdesc = soup.find(
            'div',
            class_='shortdescription nomobile noexcerpt noprint searchaux'
        )
        if shortdesc:
            text = shortdesc.get_text(" ", strip=True)
            # Capture everything up to " racing"
            m = re.match(r'^(.+?)\s+racing\b', text, re.IGNORECASE)
            if m:
                nat_string = m.group(1).strip()
                return self.normalize_nationality_text(nat_string)
            # Fallback to first word
            m2 = re.match(r'^([A-Z][a-z]+)', text)
            if m2:
                return m2.group(1).capitalize()

        return None

    def normalize_nationality_text(self, text):
        """Normalize nationality text to consistent format"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        text = re.sub(r'\s*[,\/;|]\s*', ', ', text)  # Normalize separators to commas
        text = re.sub(r'\s*[–—\-]\s*', '-', text)  # Normalize hyphens

        # Split and process individual nationality terms
        parts = []
        for part in re.split(r'[,\s]', text):
            part = part.strip()
            if not part:
                continue

            # Handle hyphenated nationalities
            if '-' in part:
                hyphenated = [p.strip().capitalize() for p in part.split('-') if p.strip()]
                if hyphenated:
                    parts.extend(hyphenated)
                continue

            parts.append(part.capitalize())

        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        return ', '.join(unique_parts)

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

        # Check for known invalid drivers
        if driver_name in self.driver_aliases and self.driver_aliases[driver_name] is None:
            print(f"Skipping known invalid driver: {driver_name}")
            profile = {
                "name": driver_name,
                "dob": None,
                "nationality": None,
                "scraped": False,
            }
            self.save_profile(profile_file, profile)
            return profile

        # Find Wikipedia page
        wiki_url = self.search_wikipedia_page(driver_name)
        if not wiki_url:
            print(f"No Wikipedia page found for {driver_name}")
            profile = {
                "name": driver_name,
                "dob": None,
                "nationality": None,
                "scraped": False
            }
            self.save_profile(profile_file, profile)
            return profile

        try:
            response = self.session.get(wiki_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')

            # Double-check this is a racing driver page
            if not self.is_racing_driver_page(soup):
                print(f"Warning: {driver_name} page may not be about racing driver")

            # Extract data
            dob = self.extract_dob(soup)
            nationality = self.extract_nationality(soup)

            profile = {
                "name": driver_name,
                "dob": dob,
                "nationality": nationality,
                "wiki_url": wiki_url,
                "scraped": True,
                "scraped_date": datetime.now().isoformat()
            }

            self.save_profile(profile_file, profile)
            del soup
            return profile

        except Exception as e:
            print(f"Error scraping {driver_name}: {e}")
            profile = {
                "name": driver_name,
                "dob": None,
                "nationality": nationality,
                "scraped": False,
                "error": str(e)
            }
            self.save_profile(profile_file, profile)
            return profile

    def extract_dob(self, soup):
        """Extract date of birth from Wikipedia page."""
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
                                return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
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

    def save_profile(self, filename, profile):
        """Save profile to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    def batch_scrape_drivers(self, driver_list):
        """Scrape profiles for list of drivers."""
        profiles = {}
        unique_drivers = list(set(driver_list))

        print(f"Scraping profiles for {len(unique_drivers)} unique drivers...")

        for i, driver in enumerate(unique_drivers, 1):
            # print(f"Progress: {i}/{len(unique_drivers)}")
            profiles[driver] = self.scrape_driver_profile(driver)

        return profiles


def get_all_drivers_from_data():
    """Extract all driver names from data files."""
    all_drivers = set()
    series_map = {
        'F1': {
            'entries_pattern': 'f1_{year}_entries.csv'
        },
        'F2': {
            'entries_pattern': 'f2_{year}_entries.csv'
        },
        'F3': {
            'entries_pattern': 'f3_{year}_entries.csv'
        },
        'F3_European': {
            'entries_pattern': 'f3_euro_{year}_entries.csv'
        }
    }

    for series, patterns in series_map.items():
        series_dirs = glob.glob(f"data/{series}/*")

        for year_dir in series_dirs:
            year = os.path.basename(year_dir)
            if not year.isdigit():
                continue

            # Get file patterns for this series
            entries_pattern = patterns['entries_pattern'].format(year=year)

            # Check entries file
            entries_file = os.path.join(year_dir, entries_pattern)
            if os.path.exists(entries_file):
                try:
                    df = pd.read_csv(entries_file)
                    if 'Driver' in df.columns:
                        drivers = df['Driver'].dropna().str.strip().unique()
                        all_drivers.update(drivers)
                except Exception as e:
                    print(f"Error reading {entries_file}: {e}")

    return sorted(list(all_drivers))


def scrape_drivers():
    print("Scanning data files for driver names...")
    all_drivers = get_all_drivers_from_data()

    if not all_drivers:
        print("No drivers found. Check your data directory structure.")
        return

    print(f"Found {len(all_drivers)} unique drivers")

    scraper = DriverProfileScraper()

    print("Starting driver profile scraping...")
    profiles = scraper.batch_scrape_drivers(all_drivers)

    scraped_count = sum(1 for p in profiles.values() if p.get('scraped', False))
    with_dob = sum(1 for p in profiles.values() if p.get('dob'))
    del profiles

    print("Scraping complete!")
    print(f"Successfully scraped: {scraped_count}/{len(all_drivers)}")
    print(f"Drivers with DOB: {with_dob}")
    print(f"Profiles saved to: {scraper.profiles_dir}/")


if __name__ == "__main__":
    scrape_drivers()
