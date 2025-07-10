import csv
import os
from bs4 import BeautifulSoup
from scraping.scraping_utils import remove_citations

HEADER_MAPPING = {
    'Entrant': 'Team',
    'Teams': 'Team',
    'Team': 'Team',
    'No.': 'No.',
    'Driver name': 'Driver',
    'Drivers': 'Driver',
    'Race Drivers': 'Driver',
    'Race drivers': 'Driver',
    'Driver': 'Driver',
    'Rounds': 'Rounds',
}
UNWANTED_COLUMNS = {'chassis', 'engine', 'status', 'constructor', 'power unit'}


def map_url(soup, year, formula, series_type="main"):
    if formula == 1:
        if year >= 2018 or year == 2016:
            return soup.find("h2", {"id": "Entries"})
        else:
            return soup.find("h2", {"id": "Teams_and_drivers"})
    elif series_type == "f3_euro":
        if year == 2018:
            return soup.find("h2", {"id": "Entries"})
        elif year < 2016:
            return soup.find("h2", {"id": "Drivers_and_teams"})
        else:
            return soup.find("h2", {"id": "Teams_and_drivers"})
    elif year == 2018 and formula == 2:
        return soup.find("h2", {"id": "Entries"})
    elif year <= 2018 or (formula == 3 and year == 2019):
        return soup.find("h2", {"id": "Teams_and_drivers"})
    else:
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
        if formula == 1 and year > 2015:
            # Two-row header structure for F1 2016+
            if len(all_rows) < 2:
                headers = [remove_citations(th.get_text(strip=True)) 
                           for th in all_rows[0].find_all("th")]
            else:
                # Process first header row
                headers_row1 = all_rows[0].find_all("th")
                combined_headers = []
                for th in headers_row1:
                    colspan = int(th.get('colspan', '1'))
                    if colspan > 1:
                        # Expand colspan into placeholders
                        combined_headers.extend([None] * colspan)
                    else:
                        text = remove_citations(th.get_text(strip=True))
                        combined_headers.append(text)
                
                # Process second header row
                headers_row2 = all_rows[1].find_all("th")
                h2_iter = iter(headers_row2)
                # Replace placeholders with actual headers
                for i in range(len(combined_headers)):
                    if combined_headers[i] is None:
                        try:
                            th = next(h2_iter)
                            text = remove_citations(th.get_text(strip=True))
                            combined_headers[i] = text
                        except StopIteration:
                            pass  # No more headers to fill
                headers = combined_headers
            data_rows = all_rows[2:]
        else:
            # Single-row header processing
            header_row = all_rows[0]
            headers = [remove_citations(header.get_text(strip=True))
                       for header in header_row.find_all("th")]
            data_rows = all_rows[1:]

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
        if series_type == "f3_euro" or formula == 1:
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
            remaining_cells = cells[cell_index:]
            if formula == 1 and year >= 2014:
                # F1 2014+ structure with multiple drivers in one row
                for cell in remaining_cells:
                    # Create a copy to avoid modifying original
                    cell_copy = BeautifulSoup(str(cell), 'lxml').find()
                    # Remove citations
                    for sup in cell_copy.find_all('sup'):
                        sup.decompose()
                    # Extract text lines
                    lines = [s.strip() for s in cell_copy.stripped_strings]
                    row_data.append(lines)
                
                # Split into separate rows per driver
                driver_count = max(len(driver_data) for driver_data in row_data[4:]) if len(row_data) > 4 else 0
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
                        row_data.append(remove_citations(cell.get_text(strip=True)))
                    else:
                        row_data.append('')

                if any("ineligible" in cell.lower() for cell in row_data):
                    continue
                
                # Remove unwanted columns
                for idx in sorted(unwanted_indices, reverse=True):
                    if idx < len(row_data):
                        del row_data[idx]
                
                writer.writerow(row_data)
