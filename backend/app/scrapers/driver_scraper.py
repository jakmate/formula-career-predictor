import glob
import json
import os
import pandas as pd
import re
from datetime import datetime

from app.config import PROFILES_DIR
from app.scrapers.scraping_utils import create_session

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DRIVER_ALIASES = {
    "Lucas di Grassi": "Lucas Di Grassi",
    "Alfonso Celis Jr.": "Alfonso Celis",
    "Keyvan Andres": "Keyvan Andres Soori",
    "Gabriel Bortoleto": "Gabriel Lourenzo Bortoleto Oliveira",
    "Gabriel Chaves": "Gabby Chaves",
    "Robert Kubica": "Robert Jozef Kubica",
    "Yifei Ye": "Ye Yifei",
    "Frederik Vesti": "Frederik Stamm Vesti"
}


def get_driver_filename(driver_name):
    """Create safe filename from driver name."""
    safe_name = re.sub(r'[^\w\s-]', '', driver_name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return f"{safe_name.lower()}.json"


def search_wikidata_drivers(driver_names, session, batch_size=100):
    """Search for racing drivers in Wikidata using batched SPARQL."""
    results = {}

    # Process in batches
    for i in range(0, len(driver_names), batch_size):
        batch = driver_names[i:i + batch_size]

        # Build VALUES clause for batch
        values_list = []
        for name in batch:
            values_list.extend([f'"{name}"@en', f'"{name}"@mul'])
        values_clause = " ".join(values_list)

        query = f"""
        SELECT ?person ?personLabel ?dob ?nationality ?nationalityLabel
            ?citizenship ?citizenshipLabel ?nameMatch WHERE {{
          VALUES ?nameMatch {{ {values_clause} }}
          {{
            ?person rdfs:label ?nameMatch .
          }} UNION {{
            ?person skos:altLabel ?nameMatch .
          }}
          ?person wdt:P106 ?occupation .
          FILTER(?occupation = wd:Q10349745 ||
            ?occupation = wd:Q378622 || ?occupation = wd:Q10841764)

          OPTIONAL {{ ?person wdt:P569 ?dob }}
          OPTIONAL {{ ?person wdt:P1532 ?nationality }}
          OPTIONAL {{ ?person wdt:P27 ?citizenship }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        """

        try:
            response = session.get(
                SPARQL_ENDPOINT,
                params={'query': query, 'format': 'json'},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                # Group results by matched name
                for binding in data['results']['bindings']:
                    name = binding['nameMatch']['value']
                    if name not in results:
                        results[name] = binding
            else:
                print(response.status_code)

        except Exception as e:
            print(f"Batch query error for batch {i//batch_size + 1}: {e}")

    return results


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


def needs_rescrape(existing_profile, new_data):
    """Check if profile needs to be rescraped based on data changes."""
    if not existing_profile.get('scraped', False):
        # If previous scrape failed, try again
        return True

    # Extract new values
    new_dob = extract_dob_from_result(new_data)
    new_nationality = extract_nationality_from_result(new_data)

    # Compare with existing values
    dob_changed = existing_profile.get('dob') != new_dob
    nationality_changed = existing_profile.get('nationality') != new_nationality

    return dob_changed or nationality_changed


def get_all_drivers_from_data():
    """Extract all driver names from data files."""
    all_drivers = set()
    series_map = {
        'F1': 'f1_{year}_entries.csv',
        'F2': 'f2_{year}_entries.csv',
        'F3': 'f3_{year}_entries.csv'
    }

    for series, pattern in series_map.items():
        series_dirs = glob.glob(f"data/{series}/*")

        for year_dir in series_dirs:
            year = os.path.basename(year_dir)
            if not year.isdigit():
                continue

            entries_file = os.path.join(year_dir, pattern.format(year=year))

            if os.path.exists(entries_file):
                try:
                    df = pd.read_csv(entries_file)
                    if 'Driver' in df.columns:
                        drivers = df['Driver'].dropna().str.strip().unique()
                        all_drivers.update(drivers)
                except Exception as e:
                    print(f"Error reading {entries_file}: {e}")

    return sorted(list(all_drivers))


def scrape_drivers(session=None):
    """Main function to scrape all driver profiles using batched queries."""
    if not session:
        session = create_session()

    print("Scanning data files for driver names...")
    all_drivers = get_all_drivers_from_data()

    if not all_drivers:
        print("No drivers found.")
        return

    print(f"Found {len(all_drivers)} unique drivers")

    # Ensure profiles directory exists
    os.makedirs(PROFILES_DIR, exist_ok=True)

    try:
        # Create mapping of original name -> search name
        driver_search_map = {}
        for driver in all_drivers:
            search_name = DRIVER_ALIASES.get(driver, driver)
            if search_name is not None:  # Skip explicitly invalid drivers
                driver_search_map[driver] = search_name

        # Separate drivers into new and existing
        new_drivers = []
        existing_drivers = []

        for driver in driver_search_map.keys():
            profile_file = os.path.join(PROFILES_DIR, get_driver_filename(driver))
            if os.path.exists(profile_file):
                existing_drivers.append(driver)
            else:
                new_drivers.append(driver)

        print(f"New drivers: {len(new_drivers)}, Existing: {len(existing_drivers)}")

        # Batch query for new drivers (use search names)
        if new_drivers:
            print(f"Querying {len(new_drivers)} new drivers in batches...")
            search_names = [driver_search_map[d] for d in new_drivers]
            new_results = search_wikidata_drivers(search_names, session)

            # Map results back to original names
            new_results_mapped = {}
            for driver in new_drivers:
                search_name = driver_search_map[driver]
                if search_name in new_results:
                    new_results_mapped[driver] = new_results[search_name]
            new_results = new_results_mapped

            for driver in new_drivers:
                result = new_results.get(driver)

                if not result:
                    print(f"NO RESULTS FOR {driver}")
                    profile = {
                        "name": driver,
                        "dob": None,
                        "nationality": None,
                        "scraped": False
                    }
                else:
                    profile = {
                        "name": driver,
                        "dob": extract_dob_from_result(result),
                        "nationality": extract_nationality_from_result(result),
                        "wikidata_id": result['person']['value'].split('/')[-1],
                        "wikidata_url": result['person']['value'],
                        "scraped": True,
                        "scraped_date": datetime.now().isoformat()
                    }

                profile_file = os.path.join(PROFILES_DIR, get_driver_filename(driver))
                save_profile(profile_file, profile)

        # Check existing drivers for updates
        if existing_drivers:
            print(f"Checking {len(existing_drivers)} existing drivers for updates...")
            search_names = [driver_search_map[d] for d in existing_drivers]
            existing_results = search_wikidata_drivers(search_names, session)

            # Map results back to original names
            existing_results_mapped = {}
            for driver in existing_drivers:
                search_name = driver_search_map[driver]
                if search_name in existing_results:
                    existing_results_mapped[driver] = existing_results[search_name]
            existing_results = existing_results_mapped

            updated_count = 0
            for driver in existing_drivers:
                profile_file = os.path.join(PROFILES_DIR, get_driver_filename(driver))
                with open(profile_file, 'r', encoding='utf-8') as f:
                    existing_profile = json.load(f)

                result = existing_results.get(driver)
                if result and needs_rescrape(existing_profile, result):
                    print(f"Updating {driver}...")
                    profile = {
                        "name": driver,
                        "dob": extract_dob_from_result(result),
                        "nationality": extract_nationality_from_result(result),
                        "wikidata_id": result['person']['value'].split('/')[-1],
                        "wikidata_url": result['person']['value'],
                        "scraped": True,
                        "scraped_date": datetime.now().isoformat()
                    }
                    save_profile(profile_file, profile)
                    updated_count += 1

            print(f"Updated {updated_count} profiles")

        print("Scraping complete")

    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover
    scrape_drivers()
