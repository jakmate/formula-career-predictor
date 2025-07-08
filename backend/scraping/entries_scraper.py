import csv
import os
from scraping.scraping_utils import remove_citations

HEADER_MAPPING = {
    'Entrant': 'Team',
    'Teams': 'Team',
    'Team': 'Team',
    'No.': 'No.',
    'Driver name': 'Driver',
    'Drivers': 'Driver',
    'Driver': 'Driver',
    'Rounds': 'Rounds',
}
UNWANTED_COLUMNS = {'chassis', 'engine', 'status'}


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
        dir_path = os.path.join("data/F3_European", str(year))
        filename = f"f3_euro_{year}_entries.csv"
    else:
        dir_path = os.path.join(f"data/F{formula}", str(year))
        filename = f"f{formula}_{year}_entries.csv"

    os.makedirs(dir_path, exist_ok=True)
    full_path = os.path.join(dir_path, filename)

    with open(full_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)

        # Process headers
        header_row = all_rows[0]
        headers = [remove_citations(header.get_text(strip=True))
                   for header in header_row.find_all("th")]
        headers = [HEADER_MAPPING.get(h, h) for h in headers]
        num_columns = len(headers)

        # Remove unwanted columns from headers
        unwanted_indices = [
            idx for idx, header in enumerate(headers)
            if header.strip().lower() in UNWANTED_COLUMNS
        ]
        # Delete indices in descending order to avoid shifting
        for idx in sorted(unwanted_indices, reverse=True):
            del headers[idx]

        writer.writerow(headers)

        # Skip header row
        data_rows = all_rows[1:]

        # Remove footer row
        if len(data_rows) > 0:
            if not (formula == 2 and year < 2018) and not (
                formula == 3 and year < 2017) and not (
                    formula == 3 and year == 2017 and series_type == 'f3_euro'):
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

            # Remove unwanted columns from this row
            for idx in sorted(unwanted_indices, reverse=True):
                if idx < len(row_data):
                    del row_data[idx]

            writer.writerow(row_data)
