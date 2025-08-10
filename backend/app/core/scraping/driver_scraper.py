import glob
import json
import os
import pandas as pd
import re
import requests
from datetime import datetime

from app.config import PROFILES_DIR

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DRIVER_ALIASES = {
    "Peter Li": "Zhi Cong Li",
    "Richard Goddard": "Spike Goddard",
    "Lucas di Grassi": "Lucas Di Grassi",
    "Andy Chang": "Andy Chang Wing Chung",
    "Alfonso Celis Jr.": "Alfonso Celis",
    "Guanyu Zhou": "Zhou Guanyu",
    "Rodolfo González": "Rodolfo Gonzalez",
    "Keyvan Andres": "Keyvan Andres Soori",
    "Gabriel Bortoleto": "Gabriel Lourenzo Bortoleto Oliveira",
    "Mitch Gilbert": "Mitchell Gilbert",
    "Gabriel Chaves": "Gabby Chaves",
    "Gabriele Minì": "Gabriele Mini",
    "Robert Kubica": "Robert Jozef Kubica",
    "Yifei Ye": "Ye Yifei"
}


def get_driver_filename(driver_name):
    """Create safe filename from driver name."""
    safe_name = re.sub(r'[^\w\s-]', '', driver_name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return f"{safe_name.lower()}.json"


def search_wikidata_driver(driver_name, session):
    """Search for racing driver in Wikidata using SPARQL."""
    # Check for known aliases first
    if driver_name in DRIVER_ALIASES:
        alias = DRIVER_ALIASES[driver_name]
        if alias is None:  # Explicitly marked as invalid
            return None
        driver_name = alias

    # SPARQL query to find racing drivers by name
    query = f"""
    SELECT ?person ?personLabel ?dob ?nationality ?nationalityLabel
        ?citizenship ?citizenshipLabel WHERE {{
      {{
        ?person rdfs:label "{driver_name}"@en .
      }} UNION {{
        ?person skos:altLabel "{driver_name}"@en .
      }}
      ?person wdt:P106 ?occupation .
      # racing automobile driver or racing driver or Formula One driver
      FILTER(?occupation = wd:Q10349745 ||
        ?occupation = wd:Q378622 || ?occupation = wd:Q10841764)

      OPTIONAL {{ ?person wdt:P569 ?dob }}
      OPTIONAL {{ ?person wdt:P1532 ?nationality }}  # country for sport
      OPTIONAL {{ ?person wdt:P27 ?citizenship }}    # country of citizenship

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}
    """

    try:
        response = session.get(
            SPARQL_ENDPOINT,
            params={'query': query, 'format': 'json'},
            headers={'User-Agent': 'DriverProfileScraper/1.0'},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data['results']['bindings']:
                return data['results']['bindings'][0]  # Return first match

    except Exception as e:
        print(f"Wikidata query error for {driver_name}: {e}")

    return None


def extract_nationality_from_result(result):
    """Extract nationality from Wikidata result, preferring country for sport."""
    # Prefer country for sport over citizenship
    if 'nationalityLabel' in result and result['nationalityLabel']['value']:
        return result['nationalityLabel']['value']
    elif 'citizenshipLabel' in result and result['citizenshipLabel']['value']:
        return result['citizenshipLabel']['value']
    return None


def extract_dob_from_result(result):
    """Extract date of birth from Wikidata result."""
    if 'dob' in result and result['dob']['value']:
        # Wikidata returns dates in ISO format, extract just the date part
        return result['dob']['value'].split('T')[0]
    return None


def save_profile(filename, profile):
    """Save profile to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def scrape_driver_profile(driver_name, session):
    """Scrape individual driver profile from Wikidata."""
    profile_file = os.path.join(PROFILES_DIR, get_driver_filename(driver_name))

    # Check if already scraped
    if os.path.exists(profile_file):
        with open(profile_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"Scraping profile for {driver_name}...")

    # Check for known invalid drivers
    if driver_name in DRIVER_ALIASES and DRIVER_ALIASES[driver_name] is None:
        print(f"Skipping known invalid driver: {driver_name}")
        profile = {
            "name": driver_name,
            "dob": None,
            "nationality": None,
            "scraped": False,
        }
        save_profile(profile_file, profile)
        return profile

    # Search Wikidata
    result = search_wikidata_driver(driver_name, session)
    if not result:
        print(f"No Wikidata entry found for {driver_name}")
        profile = {
            "name": driver_name,
            "dob": None,
            "nationality": None,
            "scraped": False
        }
        save_profile(profile_file, profile)
        return profile

    try:
        # Extract data
        dob = extract_dob_from_result(result)
        nationality = extract_nationality_from_result(result)
        wikidata_id = result['person']['value'].split('/')[-1]

        profile = {
            "name": driver_name,
            "dob": dob,
            "nationality": nationality,
            "wikidata_id": wikidata_id,
            "wikidata_url": result['person']['value'],
            "scraped": True,
            "scraped_date": datetime.now().isoformat()
        }

        save_profile(profile_file, profile)
        return profile

    except Exception as e:
        print(f"Error processing {driver_name}: {e}")
        profile = {
            "name": driver_name,
            "dob": None,
            "nationality": None,
            "scraped": False,
            "error": str(e)
        }
        save_profile(profile_file, profile)
        return profile


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

            entries_pattern = patterns['entries_pattern'].format(year=year)
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
    """Main function to scrape all driver profiles."""
    print("Scanning data files for driver names...")
    all_drivers = get_all_drivers_from_data()

    if not all_drivers:
        print("No drivers found. Check your data directory structure.")
        return

    print(f"Found {len(all_drivers)} unique drivers")

    # Ensure profiles directory exists
    os.makedirs(PROFILES_DIR, exist_ok=True)

    # Create session once for all requests
    session = requests.Session()

    try:
        print("Starting driver profile scraping...")
        unique_drivers = list(set(all_drivers))
        profiles = {}

        for i, driver in enumerate(unique_drivers, 1):
            profiles[driver] = scrape_driver_profile(driver, session)

        scraped_count = sum(1 for p in profiles.values() if p.get('scraped', False))
        print(f"Scraping complete: {scraped_count}/{len(all_drivers)}")

    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    scrape_drivers()
