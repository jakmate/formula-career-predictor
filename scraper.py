import os
import requests
from bs4 import BeautifulSoup
import csv

BASE_URL = "https://en.wikipedia.org/wiki/"


def process_championship(soup, championship_type, year, file_suffix, formula):
    # Determine heading ID based on year and championship type
    if year == 2013 and formula == 2 and championship_type == 'Drivers\'':
        heading_id = f"{championship_type}_championship"
    elif year < 2013:
        heading_id = f"{championship_type}_Championship"
    elif year < 2023:
        heading_id = f"{championship_type}_championship"
    else:
        heading_id = f"{championship_type}_Championship_standings"

    heading = soup.find("h3", {"id": heading_id.replace(" ", "_")})

    if not heading:
        print(f"No {championship_type} heading found for {year}")
        return

    table = heading.find_next("table", {"class": "wikitable"})
    if year == 2013 and formula == 2 and championship_type == 'Drivers\'':
        table = heading
        for _ in range(3):
            table = table.find_next("table", {"class": "wikitable"})
    if not table:
        print(f"No {championship_type} table found for {year}")
        return

    all_rows = table.find_all("tr")
    if len(all_rows) < 3:
        print(f"Not enough rows in {championship_type} table for {year}")
        return

    # Create directiories
    dir_path = os.path.join(f"F{formula}", str(year))
    os.makedirs(dir_path, exist_ok=True)
    filename = f"f{formula}_{year}_{file_suffix}.csv"
    full_path = os.path.join(dir_path, filename)

    with open(full_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)

        # Check if it's team or driver standings
        is_team_standings = "team" in file_suffix.lower()

        race_header_row = all_rows[0]

        # Get all header cells from both rows
        race_headers = race_header_row.find_all("th")

        if (year > 2012 and formula == 3) or (year > 2016 and formula == 2):
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

        # Processall race headers
        for i, th in enumerate(race_headers[col_index:], col_index):
            race_name = th.get_text(strip=True)
            # Stop when we hit the Points column
            if not race_name or race_name.lower() in ['points', 'pts']:
                break

            colspan = int(th.get('colspan', 1))

            # Get corresponding round headers for this race
            if year > 2016 or (year > 2012 and formula == 3):
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
        data_rows = all_rows[2:] if (year > 2012 and formula == 3) or (
            year > 2016 and formula == 2) else all_rows[1:]

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


def process_entries(soup, year, formula):
    if year == 2018 and formula == 2:
        heading = soup.find("h2", {"id": "Entries"})
    elif year <= 2018 or (formula == 3 and year == 2019):
        heading = soup.find("h2", {"id": "Teams_and_drivers"})
    else:
        heading = soup.find("h2", {"id": "Entries"})

    if not heading:
        print(f"No entries heading found for {formula} {year}")
        return

    table = heading.find_next("table", {"class": "wikitable"})
    if not table:
        print(f"No table found for {formula} {year}")
        return

    all_rows = table.find_all("tr")
    if len(all_rows) < 3:
        print(f"Not enough rows in table for {formula} {year}")
        return

    # Create directories
    dir_path = os.path.join(f"F{formula}", str(year))
    os.makedirs(dir_path, exist_ok=True)
    filename = f"f{formula}_{year}_entries.csv"
    full_path = os.path.join(dir_path, filename)

    with open(full_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)

        # Extract headers
        header_row = all_rows[0]
        headers = [header.get_text(strip=True)
                   for header in header_row.find_all("th")]
        writer.writerow(headers)

        # Data processing - skip header row
        data_rows = all_rows[1:]

        # Remove footer row (if detected)
        if len(data_rows) > 0:
            last_row = data_rows[-1]
            if len(last_row.find_all(["td", "th"])) < 3:
                data_rows = data_rows[:-1]

        # Track rowspan for Entrant
        current_entrant = ""
        entrant_rowspan = 0

        for row in data_rows:
            cells = row.find_all(["td", "th"])
            row_data = []

            # Check for new entrant
            if entrant_rowspan <= 0:
                current_entrant = ""
                if cells:
                    first_cell = cells[0]
                    rowspan = int(first_cell.get("rowspan", 1))
                    if rowspan > 1:
                        current_entrant = first_cell.get_text(strip=True)
                        entrant_rowspan = rowspan
                        cells = cells[1:]  # Remove entrant cell

            # Extract data from cells
            no = cells[0].get_text(strip=True) if len(cells) > 0 else ""
            driver = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            rounds = cells[2].get_text(strip=True) if len(cells) > 2 else ""

            row_data = [current_entrant, no, driver, rounds]
            writer.writerow(row_data)

            # Update rowspan counter (decrement after processing the row)
            entrant_rowspan = max(entrant_rowspan - 1, 0)


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
            process_championship(soup, "Drivers'", year,
                                 "drivers_standings", num)

        except Exception as e:
            print(f"Error processing {year}: {str(e)}")
