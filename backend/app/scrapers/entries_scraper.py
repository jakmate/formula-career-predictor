import csv
from app.scrapers.scraping_utils import create_output_file, remove_superscripts

HEADER_MAPPING = {
    'Entrant': 'Team',
    'Teams': 'Team',
    'Driver name': 'Driver',
    'Drivers': 'Driver',
    'Race Drivers': 'Driver',
    'Race drivers': 'Driver',
}

UNWANTED_COLUMNS = {'Chassis', 'Engine', 'Status', 'Constructor', 'Power unit'}


def get_entries_heading_id(year, series):
    """Determine the heading ID to search for based on series and year."""
    if series == 1:
        if year >= 2018 or year == 2016:
            return "Entries"
        return "Teams_and_drivers"
    if year == 2018 and series == 2:
        return "Entries"
    if year <= 2018 or (series == 3 and year == 2019):
        return "Teams_and_drivers"
    return "Entries"


def find_entries_table(soup, year, series):
    """Find and return the entries table from the soup."""
    heading_id = get_entries_heading_id(year, series)
    heading = soup.find("h2", {"id": heading_id})
    if not heading:
        return None
    return heading.find_next("table", {"class": "wikitable"})


def get_rowspan_column_count(series, year):
    """Get number of columns that use rowspan based on series and year."""
    if series == 1:
        return 6 if year <= 2013 else 4
    return 2


def should_remove_footer_row(series, year):
    """Determine if footer row should be removed based on series and year."""
    if series == 2 and year == 2017:
        return False
    if series == 3 and year < 2017:
        return False
    if series == 1 and year <= 2013:
        return False
    return True


def process_headers(all_rows, series, year):
    """Process table headers and return headers list and data rows."""
    if series == 1 and year > 2015:
        return process_multirow_headers(all_rows)
    else:
        return process_single_row_headers(all_rows)


def process_multirow_headers(all_rows):
    """Process two-row header structure for F1 2016+."""
    headers_row1 = all_rows[0].find_all("th")
    combined_headers = []

    # Build placeholder structure from first row
    for th in headers_row1:
        colspan = int(th.get('colspan', '1'))
        if colspan > 1:
            combined_headers.extend([None] * colspan)
        else:
            combined_headers.append(remove_superscripts(th))

    # Fill placeholders with second row headers
    headers_row2 = all_rows[1].find_all("th")
    h2_iter = iter(headers_row2)
    for i, header in enumerate(combined_headers):
        if header is None:
            try:
                th = next(h2_iter)
                combined_headers[i] = remove_superscripts(th)
            except StopIteration:
                break

    return combined_headers, all_rows[2:]


def process_single_row_headers(all_rows):
    """Process single-row header structure."""
    header_row = all_rows[0]
    headers = [remove_superscripts(header) for header in header_row.find_all("th")]
    return headers, all_rows[1:]


def clean_headers(headers):
    """Apply header mapping and return unwanted column indices."""
    mapped_headers = [HEADER_MAPPING.get(h, h) for h in headers]
    unwanted_indices = [
        idx for idx, header in enumerate(mapped_headers)
        if header.strip() in UNWANTED_COLUMNS
    ]

    # Remove unwanted columns
    clean_headers = mapped_headers.copy()
    for idx in sorted(unwanted_indices, reverse=True):
        del clean_headers[idx]

    return clean_headers, unwanted_indices


def remove_footer_if_needed(data_rows, num_columns, series, year):
    """Remove footer row if conditions are met."""
    if not data_rows or not should_remove_footer_row(series, year):
        return data_rows

    last_row = data_rows[-1]
    if len(last_row.find_all(["td", "th"])) < num_columns:
        return data_rows[:-1]
    return data_rows


def process_rowspan_columns(trackers, cells, rowspan_columns):
    """Process columns with rowspan and return row data and updated cell index."""
    row_data = []
    cell_index = 0

    for col_idx in range(rowspan_columns):
        if trackers[col_idx]['remaining'] > 0:
            row_data.append(trackers[col_idx]['value'])
            trackers[col_idx]['remaining'] -= 1
        else:
            if cell_index < len(cells):
                cell = cells[cell_index]
                cell_index += 1
                value = remove_superscripts(cell)
                rowspan_val = int(cell.get('rowspan', '1'))
                trackers[col_idx] = {
                    'value': value,
                    'remaining': rowspan_val - 1
                }
                row_data.append(value)
            else:
                row_data.append('')

    return row_data, cell_index


def process_f1_modern_drivers(remaining_cells):
    """Process F1 2014+ structure with multiple drivers in one row."""
    processed_cells = []

    for cell in remaining_cells:
        # Remove superscripts
        for sup in cell.find_all('sup'):
            sup.decompose()

        # Extract and merge text lines
        lines = list(cell.stripped_strings)
        merged = []
        i = 0
        while i < len(lines):
            if (i + 1 < len(lines) and lines[i].isdigit() and lines[i+1].startswith('–')):
                merged.append(lines[i] + lines[i+1])
                i += 2
            else:
                merged.append(lines[i])
                i += 1
        processed_cells.append(merged)

    return processed_cells


def write_f1_modern_rows(writer, row_data, processed_cells, unwanted_indices):
    """Write separate rows for each driver in F1 2014+ format."""
    driver_count = 0
    if len(processed_cells) > 0:
        driver_count = max(len(driver_data) for driver_data in processed_cells)

    for i in range(driver_count):
        driver_row = row_data[:4]  # Team data
        for driver_data in processed_cells:
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


def process_standard_row(cells, cell_index, row_data, num_columns,
                         driver_idx, series, unwanted_indices):
    """Process standard row structure and return final row data."""
    # Fill remaining columns
    while len(row_data) < num_columns:
        if cell_index < len(cells):
            cell = cells[cell_index]
            cell_index += 1
            row_data.append(remove_superscripts(cell))
        else:
            row_data.append('')

    # Handle special cases
    if series == 3 and row_data[driver_idx] == "Robert Visoiu":
        row_data[driver_idx] = "Robert Vișoiu"
    if series == 2 and row_data[driver_idx] == "Andrea Kimi Antonelli":
        row_data[driver_idx] = "Kimi Antonelli"
    if (series == 2 or series == 3) and row_data[driver_idx] == "Guanyu Zhou":
        row_data[driver_idx] = "Zhou Guanyu"

    # Remove unwanted columns
    final_row = row_data.copy()
    for idx in sorted(unwanted_indices, reverse=True):
        if idx < len(final_row):
            del final_row[idx]

    return final_row


def process_entries(soup, year, series):
    """Process entries table and save to CSV."""
    table = find_entries_table(soup, year, series)
    if not table:
        print(f"No entries table found for F{series} {year}")
        return

    all_rows = table.find_all("tr")
    if len(all_rows) < 3:
        print(f"Not enough rows in table for F{series} {year}")
        return

    # Process headers
    headers, data_rows = process_headers(all_rows, series, year)
    clean_headers_list, unwanted_indices = clean_headers(headers)
    num_columns = len(headers)

    # Get driver column index (cache for performance)
    driver_idx = None
    try:
        driver_idx = clean_headers_list.index("Driver")
    except ValueError:
        pass

    # Remove footer if needed
    data_rows = remove_footer_if_needed(data_rows, num_columns, series, year)

    # Setup for processing
    rowspan_columns = get_rowspan_column_count(series, year)
    trackers = [{'value': '', 'remaining': 0} for _ in range(rowspan_columns)]

    # Pre-sort unwanted indices for performance
    sorted_unwanted_indices = sorted(unwanted_indices, reverse=True)

    # Write to file
    filename = f"f{series}_{year}_entries.csv"
    full_path = create_output_file(series, year, filename)

    with open(full_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(clean_headers_list)

        for row in data_rows:
            cells = row.find_all(["td", "th"])

            # Process rowspan columns
            row_data, cell_index = process_rowspan_columns(
                trackers, cells, rowspan_columns
            )

            remaining_cells = cells[cell_index:]

            if series == 1 and year >= 2014:
                # F1 2014+ multi-driver structure
                processed_cells = process_f1_modern_drivers(remaining_cells)
                row_data.extend(processed_cells)
                write_f1_modern_rows(writer, row_data, processed_cells, sorted_unwanted_indices)
            else:
                # Standard structure
                final_row = process_standard_row(
                    cells, cell_index, row_data, num_columns,
                    driver_idx, series, sorted_unwanted_indices
                )
                writer.writerow(final_row)
