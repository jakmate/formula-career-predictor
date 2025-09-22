import csv
import os
from app.config import DATA_DIR
from app.core.scraping.scraping_utils import remove_superscripts


def map_url(championship_type, series, year):
    # Determine heading ID based on year and championship type
    if series == 1:
        if championship_type == 'Drivers\'':
            return "World_Drivers\'_Championship_standings"
        elif championship_type == 'Teams\'':
            return "World_Constructors\'_Championship_standings"
    elif year == 2013 and series == 2 and championship_type == 'Drivers\'':
        return f"{championship_type}_championship"
    elif year < 2013:
        return f"{championship_type}_Championship"
    elif year < 2023:
        return f"{championship_type}_championship"
    return f"{championship_type}_Championship_standings"


def process_championship(soup, championship_type, year, file_suffix, series):
    heading_id = map_url(championship_type, series, year)
    heading = soup.find("h3", {"id": heading_id.replace(" ", "_")})

    if not heading:
        print(f"No {championship_type} heading found for {year} {series}")
        return

    table = heading.find_next("table", {"class": "wikitable"})
    if year == 2013 and series == 2 and championship_type == 'Drivers\'':
        table = heading
        for _ in range(3):
            table = table.find_next("table", {"class": "wikitable"})
    if not table:
        print(f"No {championship_type} table found for {year} {series}")
        return

    all_rows = table.find_all("tr")
    if len(all_rows) < 3:
        print(f"Not enough rows in {championship_type} {year} {series}")
        return

    dir_path = os.path.join(DATA_DIR, f"F{series}", str(year))
    filename = f"f{series}_{year}_{file_suffix}.csv"
    os.makedirs(dir_path, exist_ok=True)
    full_path = os.path.join(dir_path, filename)

    with open(full_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        race_header_row = all_rows[0]

        # Get all header cells from both rows
        race_headers = race_header_row.find_all("th")

        # Header structure
        if (year > 2012 and series == 3) or (year > 2016 and series == 2):
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
            if (year > 2012 and series == 3) or (year > 2016 and series == 2):
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
        if (year > 2012 and series == 3) or (year > 2016 and series == 2):
            data_rows = all_rows[2:]
        else:
            data_rows = all_rows[1:]

        # Remove footer rows (sources/notes)
        if ((year < 2013 and series == 3) or (series == 2 and year < 2017) or series == 1):
            if len(data_rows) > 2:
                data_rows = data_rows[:-2]
        elif championship_type == "Drivers'" and year == 2020 and series == 3:
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
