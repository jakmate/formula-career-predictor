import csv
import os
import re
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://en.wikipedia.org/wiki/"


def process_championship(
        soup,
        championship_type,
        year,
        file_suffix,
        formula,
        series_type="main"):
    # Determine heading ID based on year and championship type
    if series_type == "f3_euro":
        if year == 2012:
            if championship_type == 'Teams\'':
                return
            heading_id = "Championship_standings"
        elif year == 2013 and championship_type == 'Teams\'':
            heading_id = "Ravenol_Team_Trophy"
        else:
            heading_id = f"{championship_type}_championship"
    elif year == 2013 and formula == 2 and championship_type == 'Drivers\'':
        heading_id = f"{championship_type}_championship"
    elif year < 2013:
        heading_id = f"{championship_type}_Championship"
    elif year < 2023:
        heading_id = f"{championship_type}_championship"
    else:
        heading_id = f"{championship_type}_Championship_standings"

    if series_type == "f3_euro" and year == 2012:
        heading = soup.find("h2", {"id": heading_id})
    else:
        heading = soup.find("h3", {"id": heading_id.replace(" ", "_")})

    if not heading:
        print(f"No {championship_type} heading found for {year} {series_type}")
        return

    table = heading.find_next("table", {"class": "wikitable"})
    if year == 2013 and formula == 2 and championship_type == 'Drivers\'':
        table = heading
        for _ in range(3):
            table = table.find_next("table", {"class": "wikitable"})
    if not table:
        print(f"No {championship_type} table found for {year} {series_type}")
        return

    all_rows = table.find_all("tr")
    if len(all_rows) < 3:
        print(f"Not enough rows in {championship_type} {year} {series_type}")
        return

    # Use different folder for F3 European
    if series_type == "f3_euro":
        dir_path = os.path.join("F3_European", str(year))
        filename = f"f3_euro_{year}_{file_suffix}.csv"
    else:
        dir_path = os.path.join(f"F{formula}", str(year))
        filename = f"f{formula}_{year}_{file_suffix}.csv"

    os.makedirs(dir_path, exist_ok=True)
    full_path = os.path.join(dir_path, filename)

    with open(full_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)

        # Handle F3 European simple format (Pos, Team/Driver, Points only)
        if series_type == "f3_euro" and championship_type == "Teams'":
            # Simple 3-column format for F3 European teams
            writer.writerow(["Pos", "Team", "Points"])

            # Skip header row and footer rows
            data_rows = all_rows[1:]

            for row in data_rows:
                cells = row.find_all(["th", "td"])
                if len(cells) >= 3:
                    pos = cells[0].get_text(strip=True)
                    team = cells[1].get_text(strip=True)
                    points = cells[2].get_text(strip=True)

                    # Skip rows that aren't actual standings (like "Guest team
                    # ineligible")
                    if pos and not pos.lower().startswith('guest'):
                        writer.writerow([pos, team, points])
            return

        # Check if it's team or driver standings
        is_team_standings = "team" in file_suffix.lower()

        race_header_row = all_rows[0]

        # Get all header cells from both rows
        race_headers = race_header_row.find_all("th")

        # F3 European uses different header structure
        if series_type == "f3_euro" or (
                year > 2012 and formula == 3) or (
                year > 2016 and formula == 2):
            round_header_row = all_rows[1]
            round_headers = round_header_row.find_all("th")

        has_no_column = False
        if len(race_headers) > 2:
            second_header = race_headers[2].get_text(strip=True).lower()
            if "no." in second_header:
                has_no_column = True

        # Start building headers - skip Pos and Driver/Team columns
        combined_headers = ["Pos", "Team" if is_team_standings else "Driver"]

        # Start after Pos, Driver/Team (and No.)
        col_index = 3 if has_no_column else 2

        # Process all race headers
        for i, th in enumerate(race_headers[col_index:], col_index):
            race_name = th.get_text(strip=True)
            # Stop when we hit the Points column
            if not race_name or race_name.lower() in ['points', 'pts']:
                break

            colspan = int(th.get('colspan', 1))

            # Get corresponding round headers for this race
            if series_type == "f3_euro" or year > 2016 or (
                    year > 2012 and formula == 3):
                race_rounds = []
                round_start_idx = col_index + i - col_index - 2
                for j in range(colspan):
                    round_idx = round_start_idx + j
                    if round_idx < len(round_headers):
                        round_name = round_headers[round_idx].get_text(
                            strip=True)
                        if round_name:
                            race_rounds.append(round_name)
            else:
                race_rounds = [f"Race {r+1}" for r in range(colspan)]

            race_rounds.sort(
                key=lambda x: int(x.replace('R', ''))
                if x.replace('R', '').isdigit() else 999
            )

            # Add to headers
            for round_name in race_rounds:
                combined_headers.append(f"{race_name} {round_name}")

            col_index += colspan

        combined_headers.append("Points")
        writer.writerow(combined_headers)

        # Data processing - skip header rows and footer rows
        data_rows = all_rows[2:] if (
                            series_type == "f3_euro" or
                            (year > 2012 and formula == 3) or
                            (year > 2016 and formula == 2)) else all_rows[1:]

        # Remove footer rows (usually last 2-3 rows contain sources/notes)
        if len(data_rows) > 3:
            data_rows = data_rows[:-3]

        # Calculate the number of race columns expected
        num_race_columns = len(combined_headers) - 3  # Pos, Team, Points

        # Track rowspan values for position, team/driver, and points columns
        pos_rowspan = 0
        team_rowspan = 0
        points_rowspan = 0
        current_pos = ""
        current_team = ""
        current_points = ""

        for row in data_rows:
            cells = row.find_all(["th", "td"])
            if len(cells) < 3:  # Skip rows with insufficient data
                continue

            row_data = []
            cell_index = 0

            # Handle position column with rowspan
            if pos_rowspan <= 0:
                pos_cell = cells[cell_index]
                current_pos = pos_cell.get_text(strip=True)
                pos_rowspan = int(pos_cell.get('rowspan', 1))
                cell_index += 1
            pos_rowspan -= 1
            row_data.append(current_pos)

            # Handle team/driver column with rowspan
            if team_rowspan <= 0:
                team_cell = cells[cell_index]
                current_team = team_cell.get_text(strip=True)
                team_rowspan = int(team_cell.get('rowspan', 1))
                cell_index += 1
            team_rowspan -= 1
            row_data.append(current_team)

            # Skip No. column if it exists
            if has_no_column:
                if cell_index < len(cells):
                    cell_index += 1  # Skip the No. column

            # Calculate the expected number of race columns and slice cells
            race_cells = cells[cell_index: cell_index + num_race_columns]
            # Ensure we don't exceed the available cells
            race_cells = race_cells[:num_race_columns]

            for cell in race_cells:
                text = cell.get_text(strip=True)
                row_data.append(text)

            # Pad with empty strings if fewer race cells than expected
            while len(row_data) < len(combined_headers) - 1:  # -1 for Points
                row_data.append("")

            # Handle points column with rowspan
            if points_rowspan <= 0:
                # Find the points cell in different position than the first row
                # Points cell only present in the first row of the team
                # Search starting from the end of the race cells
                if cell_index + num_race_columns < len(cells):
                    points_cell = cells[cell_index + num_race_columns]
                    current_points = points_cell.get_text(strip=True)
                    points_rowspan = int(points_cell.get('rowspan', 1))
                else:
                    # If points cell not found, retain current_points
                    pass
            points_rowspan -= 1
            row_data.append(current_points)

            # Ensure the row has the correct number of columns
            if len(row_data) != len(combined_headers):
                # Truncate or pad as needed
                row_data = row_data[:len(combined_headers)]
                while len(row_data) < len(combined_headers):
                    row_data.append("")

            # Write the row
            writer.writerow(row_data)


def remove_citations(text):
    """Remove Wikipedia-style citations (e.g., [1], [a], [Note]) from text."""
    return re.sub(r'\[\w+\]', '', text)


def process_entries(soup, year, formula, series_type="main"):
    # Determine heading based on series type and year
    if series_type == "f3_euro":
        if year == 2018:
            heading = soup.find("h2", {"id": "Entries"})
        elif year < 2016:
            heading = soup.find("h2", {"id": "Drivers_and_teams"})
        else:
            heading = soup.find("h2", {"id": "Teams_and_drivers"})
    elif year == 2018 and formula == 2:
        heading = soup.find("h2", {"id": "Entries"})
    elif year <= 2018 or (formula == 3 and year == 2019):
        heading = soup.find("h2", {"id": "Teams_and_drivers"})
    else:
        heading = soup.find("h2", {"id": "Entries"})

    if not heading:
        print(f"No entries heading found for F{formula} {year} {series_type}")
        return

    table = heading.find_next("table", {"class": "wikitable"})
    if not table:
        print(f"No table found for F{formula} {year} {series_type}")
        return

    all_rows = table.find_all("tr")
    if len(all_rows) < 3:
        print(f"Not enough rows in table for F{formula} {year} {series_type}")
        return

    # Create directories
    if series_type == "f3_euro":
        dir_path = os.path.join("F3_European", str(year))
        filename = f"f3_euro_{year}_entries.csv"
    else:
        dir_path = os.path.join(f"F{formula}", str(year))
        filename = f"f{formula}_{year}_entries.csv"

    os.makedirs(dir_path, exist_ok=True)
    full_path = os.path.join(dir_path, filename)

    with open(full_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)

        # Extract headers
        header_row = all_rows[0]
        headers = [header.get_text(strip=True)
                   for header in header_row.find_all("th")]
        writer.writerow(headers)
        num_columns = len(headers)

        # Data processing - skip header row
        data_rows = all_rows[1:]

        # Remove footer row (if detected)
        if len(data_rows) > 0:
            last_row = data_rows[-1]
            if len(last_row.find_all(["td", "th"])) < num_columns:
                data_rows = data_rows[:-1]

        # Determine rowspan columns based on series and year
        if series_type == "f3_euro" and 2012 <= year <= 2018:
            rowspan_columns = 4  # Team, Chassis, Engine, No.
        else:
            rowspan_columns = 2  # Entrant/Team, No.

        # Initialize rowspan trackers
        trackers = [
            {'value': '', 'remaining': 0}
            for _ in range(rowspan_columns)
        ]

        for row in data_rows:
            cells = row.find_all(["td", "th"])
            row_data = []
            cell_index = 0

            # Process rowspan columns
            for col_idx in range(rowspan_columns):
                if trackers[col_idx]['remaining'] > 0:
                    # Reuse value from tracker
                    row_data.append(trackers[col_idx]['value'])
                    trackers[col_idx]['remaining'] -= 1
                else:
                    # Get new value from cell
                    if cell_index < len(cells):
                        cell = cells[cell_index]
                        cell_index += 1
                        value = remove_citations(cell.get_text(strip=True))
                        # Get rowspan value (default 1 if missing/invalid)
                        rowspan_attr = cell.get('rowspan', '1')
                        try:
                            rowspan_val = int(rowspan_attr)
                        except ValueError:
                            rowspan_val = 1
                        # Update tracker
                        trackers[col_idx] = {
                            'value': value,
                            'remaining': rowspan_val - 1
                        }
                        row_data.append(value)
                    else:
                        row_data.append('')

            # Process remaining columns (no rowspan)
            while len(row_data) < num_columns:
                if cell_index < len(cells):
                    cell = cells[cell_index]
                    cell_index += 1
                    row_data.append(remove_citations(cell.get_text(strip=True)))
                else:
                    row_data.append('')

            writer.writerow(row_data)


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
    rows = table.find_all("tr")[1:]  # Skip header row

    for row in rows:
        cells = row.find_all(["td", "th"])
        # Look for Report column (usually last column)
        for cell in cells:
            links = cell.find_all("a")
            for link in links:
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
            # Handle Monte Carlo with two groups
            return process_monte_carlo_qualifying(group_a_head, group_b_head, round_info, race_url)
        else:
            # Handle normal qualifying with single table
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
            group_a_data = extract_table_data(group_a_table)

        # Process Group B
        group_b_table = group_b_head.find_next("table", {"class": "wikitable"})
        group_b_data = []
        if group_b_table:
            group_b_data = extract_table_data(group_b_table)

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

        # Sort by position (first column should be position)
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
        # Find the qualifying table
        table = qualifying_heading.find_next("table", {"class": "wikitable"})
        if not table:
            print(f"No qualifying table found for {race_url}")
            return None

        table_data = extract_table_data(table)
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
        print(f"Error processing single qualifying table from {race_url}: {str(e)}")
        return None


def extract_table_data(table):
    """Extract data from a qualifying table"""
    try:
        all_rows = table.find_all("tr")
        if len(all_rows) < 2:
            return None

        # Get headers
        header_row = all_rows[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

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

            if row_data:  # Only add non-empty rows
                # Process grid column (last column) to max 2 digits
                grid_value = row_data[-1]
                if grid_value.isdigit() and len(grid_value) > 2:
                    row_data[-1] = grid_value[:2]  # Truncate to first two digits
                data_rows.append(row_data)

        return {
            'headers': headers,
            'data': data_rows
        }

    except Exception as e:
        print(f"Error extracting table data: {str(e)}")
        return None


def save_qualifying_data(qualifying_results, year, formula):
    """Save all qualifying data to CSV files"""
    if not qualifying_results:
        return

    dir_path = os.path.join(f"F{formula}", str(year), "qualifying")
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

            # Write headers and data
            writer.writerow(result['headers'])
            writer.writerows(result['data'])


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

                # Process qualifying data
                print(f"Processing qualifying data for F{num} {year}...")
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


if __name__ == "__main__":
    scrape()
