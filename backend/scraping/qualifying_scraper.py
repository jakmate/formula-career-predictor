import csv
import os
import requests
from bs4 import BeautifulSoup
from scraping.scraping_utils import remove_citations

COLUMN_MAPPING = {
    'Name': 'Driver',
    'Entrant': 'Team',
}


def extract_race_report_links(soup):
    """Extract race report links from the season summary table"""
    # Find Season summary heading
    season_heading = (soup.find("h3", {"id": "Season_summary"}) or
                      soup.find("h3", {"id": "Summary"}) or
                      soup.find("h2", {"id": "Results"}))

    if not season_heading:
        print("No season summary table found")
        return []

    table = season_heading.find_next("table", {"class": "wikitable"})
    if not table:
        print("No season summary table found")
        return []

    race_links = []
    for row in table.find_all("tr")[1:]:
        # Look for Report column (usually last column)
        for cell in row.find_all(["td", "th"]):
            for link in cell.find_all("a"):
                if link.get_text(strip=True).lower() == "report":
                    href = link.get("href")
                    if href and href.startswith("/wiki/"):
                        race_links.append("https://en.wikipedia.org" + href)
                    break

    return race_links


def process_qualifying_data(race_url, round_info):
    """Process qualifying data from a race report page"""
    try:
        response = requests.get(race_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find Qualifying heading
        qualifying_heading = (soup.find("h3", {"id": "Qualifying"}) or
                              soup.find("h3", {"id": "Qualifying_classification"}) or
                              soup.find("h2", {"id": "Qualifying"}))
        if not qualifying_heading:
            print(f"No qualifying section found for {race_url}")
            return None

        # Check if this is Monte Carlo with Group A and Group B
        group_a_head = soup.find("h4", {"id": "Group_A"}) or soup.find("dt", string="Group A")
        group_b_head = soup.find("h4", {"id": "Group_B"}) or soup.find("dt", string="Group B")

        if group_a_head and group_b_head:
            return process_monte_carlo_qualifying(group_a_head, group_b_head, round_info, race_url)
        else:
            return process_single_qualifying_table(qualifying_heading, round_info, race_url)

    except Exception as e:
        print(f"Error processing qualifying data from {race_url}: {str(e)}")
        return None


def process_monte_carlo_qualifying(group_a_head, group_b_head, round_info, race_url):
    """Process Monte Carlo qualifying with Group A and Group B"""
    try:
        # Process Group A
        group_a_table = group_a_head.find_next("table", {"class": "wikitable"})
        group_a_data = []
        if group_a_table:
            group_a_data = extract_quali_table_data(group_a_table)

        # Process Group B
        group_b_table = group_b_head.find_next("table", {"class": "wikitable"})
        group_b_data = []
        if group_b_table:
            group_b_data = extract_quali_table_data(group_b_table)

        if not group_a_data and not group_b_data:
            print(f"No qualifying data found in either group for {race_url}")
            return None

        # Get headers from the first available table
        headers = []
        if group_a_data:
            headers = group_a_data['headers']
        elif group_b_data:
            headers = group_b_data['headers']

        # Combine data from both groups
        combined_data = []
        if group_a_data:
            combined_data.extend(group_a_data['data'])
        if group_b_data:
            combined_data.extend(group_b_data['data'])

        # Sort by position
        try:
            combined_data.sort(key=lambda x: int(x[-1]) if x[-1].isdigit() else float('inf'))
        except (ValueError, IndexError):
            # If sorting fails, keep original order
            pass

        return {
            'headers': headers,
            'data': combined_data,
            'round_info': round_info,
            'url': race_url,
            'qualifying_type': 'Monte Carlo Groups'
        }

    except Exception as e:
        print(f"Error processing Monte Carlo qualifying from {race_url}: {str(e)}")
        return None


def process_single_qualifying_table(qualifying_heading, round_info, race_url):
    """Process standard single qualifying table"""
    try:
        table = qualifying_heading.find_next("table", {"class": "wikitable"})
        if not table:
            print(f"No qualifying table found for {race_url}")
            return None

        table_data = extract_quali_table_data(table)
        if not table_data:
            return None

        return {
            'headers': table_data['headers'],
            'data': table_data['data'],
            'round_info': round_info,
            'url': race_url,
            'qualifying_type': 'Standard'
        }

    except Exception as e:
        print(f"Error processing single qualifying table: {race_url}: {str(e)}")
        return None


def extract_quali_table_data(table):
    """Extract data from a qualifying table"""
    try:
        all_rows = table.find_all("tr")
        if len(all_rows) < 2:
            return None

        # Get headers
        header_row = all_rows[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
        headers = [COLUMN_MAPPING.get(h, h) for h in headers]

        # Identify Driver column index
        driver_col_index = None
        for idx, header in enumerate(headers):
            if header.lower() in ["driver", "name"]:
                driver_col_index = idx
                break

        # Get data rows
        data_rows = []
        for row in all_rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 4:  # Need at least Pos, No, Driver, Team
                continue

            row_data = []
            for cell in cells:
                # Clean up cell text, remove flag icons and links
                text = cell.get_text(strip=True)
                row_data.append(text)

            if row_data:
                if driver_col_index is not None and len(row_data) > driver_col_index:
                    row_data[driver_col_index] = remove_citations(row_data[driver_col_index])

                # Process grid column (last column) truncation
                grid_value = row_data[-1]
                if grid_value.isdigit() and len(grid_value) > 2:
                    row_data[-1] = grid_value[:2]  # Truncate to first two digits
                data_rows.append(row_data)

        return {'headers': headers, 'data': data_rows}

    except Exception as e:
        print(f"Error extracting table data: {str(e)}")
        return None


def save_qualifying_data(qualifying_results, year, formula):
    """Save all qualifying data to CSV files"""
    if not qualifying_results:
        return

    dir_path = os.path.join(f"data/F{formula}", str(year), "qualifying")
    base_filename = f"f{formula}_{year}_qualifying"
    os.makedirs(dir_path, exist_ok=True)

    # Save each race's qualifying data separately
    for i, result in enumerate(qualifying_results, 1):
        if result is None:
            continue

        filename = f"{base_filename}_round_{i:02d}.csv"
        full_path = os.path.join(dir_path, filename)

        with open(full_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(result['headers'])
            writer.writerows(result['data'])


def scrape_quali(soup, year, num):
    if num == 1:
        return
    race_links = extract_race_report_links(soup)
    if race_links:
        quali_results = []
        for i, link in enumerate(race_links, 1):
            result = process_qualifying_data(link, f"Round {i}")
            quali_results.append(result)

        save_qualifying_data(quali_results, year, num)
        print(f"Saved {len([r for r in quali_results if r])} qualifying sessions")
    else:
        print(f"No race report links found for F{num} {year}")
