import csv
import os
from app.config import DATA_DIR
from app.core.scraping.scraping_utils import remove_superscripts


def map_url(championship_type, formula, year, series_type="main"):
    # Determine heading ID based on year and championship type
    if formula == 1:
        if championship_type == 'Drivers\'':
            return "World_Drivers\'_Championship_standings"
        elif championship_type == 'Teams\'':
            return "World_Constructors\'_Championship_standings"
    elif series_type == "f3_euro":
        if year == 2012:
            if championship_type == 'Teams\'':
                return ""
            return "Championship_standings"
        elif year == 2013 and championship_type == 'Teams\'':
            return "Ravenol_Team_Trophy"
        return f"{championship_type}_championship"
    elif year == 2013 and formula == 2 and championship_type == 'Drivers\'':
        return f"{championship_type}_championship"
    elif year < 2013:
        return f"{championship_type}_Championship"
    elif year < 2023:
        return f"{championship_type}_championship"
    return f"{championship_type}_Championship_standings"


def process_championship(soup, championship_type, year,
                         file_suffix, formula, series_type="main"):
    heading_id = map_url(championship_type, formula, year, series_type)

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
        dir_path = os.path.join(DATA_DIR, "F3_European", str(year))
        filename = f"f3_euro_{year}_{file_suffix}.csv"
    else:
        dir_path = os.path.join(DATA_DIR, f"F{formula}", str(year))
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
                    pos = remove_superscripts(cells[0])
                    team = remove_superscripts(cells[1])
                    points = remove_superscripts(cells[2])

                    # Skip rows that aren't actual standings (like "Guest team
                    # ineligible")
                    if pos and not pos.startswith('Guest'):
                        writer.writerow([pos, team, points])
            return

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
            second_header = remove_superscripts(race_headers[2])
            if "No." in second_header or ("No" == second_header and year == 2010):
                has_no_column = True

        # Start building headers - skip Pos and Driver/Team columns
        combined_headers = ["Pos", "Team" if "team" in file_suffix.lower() else "Driver"]

        # Start after Pos, Driver/Team (and No.)
        col_index = 3 if has_no_column else 2

        # Process all race headers
        for i, th in enumerate(race_headers[col_index:], col_index):
            race_name = remove_superscripts(th, False)
            # Stop when we hit the Points column
            if not race_name or race_name in ['Points', 'Pts']:
                break

            colspan = int(th.get('colspan', 1))

            # Get corresponding round headers for this race
            if series_type == "f3_euro" or (
                year > 2012 and formula == 3) or (
                    year > 2016 and formula == 2):
                race_rounds = []
                round_start_idx = col_index + i - col_index - 2
                for j in range(colspan):
                    round_idx = round_start_idx + j
                    if round_idx < len(round_headers):
                        round_name = remove_superscripts(round_headers[round_idx], False)
                        if round_name:
                            race_rounds.append(round_name)
            else:
                race_rounds = [f"R{r+1}" for r in range(colspan)]

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
        if ((series_type == "f3_euro" and championship_type == "Drivers'") or
                year == 2025 or (series_type == "main" and year < 2013 and formula == 3)
                or formula == 1):
            if len(data_rows) > 2:
                data_rows = data_rows[:-2]
        elif formula == 2 and year < 2017:
            if len(data_rows) > 1:
                data_rows = data_rows[:-1]
        elif championship_type == "Drivers'" and year == 2020 and formula == 3:
            if len(data_rows) > 4:
                data_rows = data_rows[:-4]
        else:
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
                current_pos = remove_superscripts(pos_cell)
                pos_rowspan = int(pos_cell.get('rowspan', 1))
                cell_index += 1
            pos_rowspan -= 1
            row_data.append(current_pos)

            # Handle team/driver column with rowspan
            if team_rowspan <= 0:
                team_cell = cells[cell_index]
                current_team = remove_superscripts(team_cell)
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
                text = remove_superscripts(cell, False)
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
                    current_points = remove_superscripts(points_cell)
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
