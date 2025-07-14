import csv
import os
from app.config import DATA_DIR
from app.core.scraping.scraping_utils import remove_superscripts

HEADER_MAPPING = {
    'Entrant': 'Team',
    'Teams': 'Team',
    'Driver name': 'Driver',
    'Drivers': 'Driver',
    'Race Drivers': 'Driver',
    'Race drivers': 'Driver',
}

UNWANTED_COLUMNS = {'Chassis', 'Engine', 'Status', 'Constructor', 'Power unit'}


def map_url(soup, year, formula, series_type="main"):
    if formula == 1:
        if year >= 2018 or year == 2016:
            return soup.find("h2", {"id": "Entries"})
        return soup.find("h2", {"id": "Teams_and_drivers"})
    if series_type == "f3_euro":
        if year == 2018:
            return soup.find("h2", {"id": "Entries"})
        if year < 2016:
            return soup.find("h2", {"id": "Drivers_and_teams"})
        return soup.find("h2", {"id": "Teams_and_drivers"})
    if year == 2018 and formula == 2:
        return soup.find("h2", {"id": "Entries"})
    if year <= 2018 or (formula == 3 and year == 2019):
        return soup.find("h2", {"id": "Teams_and_drivers"})
    return soup.find("h2", {"id": "Entries"})


def process_entries(soup, year, formula, series_type="main"):
    # Determine heading based on series type and year
    heading = map_url(soup, year, formula, series_type)
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
        dir_path = os.path.join(DATA_DIR, "F3_European", str(year))
        filename = f"f3_euro_{year}_entries.csv"
    else:
        dir_path = os.path.join(DATA_DIR, f"F{formula}", str(year))
        filename = f"f{formula}_{year}_entries.csv"

    os.makedirs(dir_path, exist_ok=True)
    full_path = os.path.join(dir_path, filename)

    with open(full_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)

        # Process headers
        if formula == 1 and year > 2015:
            # Two-row header structure for F1 2016+
            headers_row1 = all_rows[0].find_all("th")
            combined_headers = []
            for th in headers_row1:
                colspan = int(th.get('colspan', '1'))
                if colspan > 1:
                    # Expand colspan into placeholders
                    combined_headers.extend([None] * colspan)
                else:
                    text = remove_superscripts(th)
                    combined_headers.append(text)

            headers_row2 = all_rows[1].find_all("th")
            h2_iter = iter(headers_row2)
            # Replace placeholders with actual headers
            for i in range(len(combined_headers)):
                if combined_headers[i] is None:
                    try:
                        th = next(h2_iter)
                        text = remove_superscripts(th)
                        combined_headers[i] = text
                    except StopIteration:
                        pass  # No more headers to fill
            headers = combined_headers
            data_rows = all_rows[2:]
        else:
            # Single-row header processing
            header_row = all_rows[0]
            headers = [remove_superscripts(header)
                       for header in header_row.find_all("th")]
            data_rows = all_rows[1:]

        headers = [HEADER_MAPPING.get(h, h) for h in headers]
        num_columns = len(headers)

        # Remove unwanted columns from headers
        unwanted_indices = [
            idx for idx, header in enumerate(headers)
            if header.strip() in UNWANTED_COLUMNS
        ]
        # Delete indices in descending order to avoid shifting
        for idx in sorted(unwanted_indices, reverse=True):
            del headers[idx]

        writer.writerow(headers)

        # Remove footer row
        if len(data_rows) > 0:
            if not (formula == 2 and year < 2018) and not (
                formula == 3 and year < 2017) and not (
                    year == 2017 and series_type == 'f3_euro') and not (
                        formula == 1 and year <= 2013
                    ):
                last_row = data_rows[-1]
                if len(last_row.find_all(["td", "th"])) < num_columns:
                    data_rows = data_rows[:-1]

        # Determine rowspan columns based on series and year
        if series_type == "f3_euro":
            rowspan_columns = 4  # Team, Chassis, Engine, No.
        elif formula == 1:
            if year <= 2013:
                rowspan_columns = 6  # Team, Constructor, Chassis, Engine, No.
            else:
                rowspan_columns = 4  # Team, Chassis, Engine, No.
        else:
            rowspan_columns = 2  # Entrant/Team, No.

        # Initialize rowspan trackers
        trackers = [
            {'value': '', 'remaining': 0}
            for _ in range(rowspan_columns)
        ]

        driver_idx = headers.index("Driver")

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
                        value = remove_superscripts(cell)
                        # Get rowspan value (default 1 if missing/invalid)
                        rowspan_val = int(cell.get('rowspan', '1'))
                        # Update tracker
                        trackers[col_idx] = {
                            'value': value,
                            'remaining': rowspan_val - 1
                        }
                        row_data.append(value)
                    else:
                        row_data.append('')

            # Process remaining columns (no rowspan)
            remaining_cells = cells[cell_index:]
            if formula == 1 and year >= 2014:
                # F1 2014+ structure with multiple drivers in one row
                for cell in remaining_cells:
                    # Remove superscripts
                    for sup in cell.find_all('sup'):
                        sup.decompose()
                    # Extract text lines
                    lines = list(cell.stripped_strings)
                    merged = []
                    i = 0
                    while i < len(lines):
                        if lines[i].isdigit() and i+1 < len(lines) and lines[i+1].startswith('–'):
                            merged.append(lines[i] + lines[i+1])
                            i += 2
                        else:
                            merged.append(lines[i])
                            i += 1
                    lines = merged
                    row_data.append(lines)

                # Split into separate rows per driver
                driver_count = 0
                if len(row_data) > 4:
                    driver_count = max(len(driver_data) for driver_data in row_data[4:])
                for i in range(driver_count):
                    driver_row = row_data[:4]  # Team data
                    for driver_data in row_data[4:]:
                        if i < len(driver_data):
                            driver_row.append(driver_data[i])
                        else:
                            driver_row.append('')

                    # Remove unwanted columns
                    final_row = driver_row.copy()
                    for idx in sorted(unwanted_indices, reverse=True):
                        if idx < len(final_row):
                            del final_row[idx]

                    writer.writerow(final_row)
            else:
                # Normal structure
                while len(row_data) < num_columns:
                    if cell_index < len(cells):
                        cell = cells[cell_index]
                        cell_index += 1
                        row_data.append(remove_superscripts(cell))
                    else:
                        row_data.append('')

                if formula == 3:
                    if series_type == 'f3_euro':
                        if any("ineligible" in cell for cell in row_data):
                            continue
                    if series_type == 'main':
                        if driver_idx is not None and row_data[driver_idx] == "Robert Visoiu":
                            row_data[driver_idx] = "Robert Vișoiu"

                # Remove unwanted columns
                for idx in sorted(unwanted_indices, reverse=True):
                    if idx < len(row_data):
                        del row_data[idx]

                writer.writerow(row_data)
