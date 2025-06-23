import csv
import os
from .scraping_utils import remove_citations


def process_championship(soup, championship_type, year,
                         file_suffix, formula, series_type="main"):
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
        dir_path = os.path.join("data/F3_European", str(year))
        filename = f"f3_euro_{year}_{file_suffix}.csv"
    else:
        dir_path = os.path.join(f"data/F{formula}", str(year))
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
            race_name = remove_citations(race_name)
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

        # Remove footer rows (sources/notes)
        if (series_type == "f3_euro" and championship_type == "Drivers'") or year == 2025:
            # For F3 European drivers standings, remove only 2 footer rows
            if len(data_rows) > 2:
                data_rows = data_rows[:-2]
        elif championship_type == "Drivers'" and year == 2020 and formula == 3:
            # For F3 2020, remove 4 rows
            if len(data_rows) > 4:
                data_rows = data_rows[:-4]
        else:
            # Default: remove 3 footer rows for other standings
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
