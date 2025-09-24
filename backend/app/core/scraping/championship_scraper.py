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


def find_championship_table(soup, championship_type, series, year):
    """Find and return the championship table from the soup."""
    heading_id = map_url(championship_type, series, year)
    heading = soup.find("h3", {"id": heading_id.replace(" ", "_")})
    
    if not heading:
        return None, f"No {championship_type} heading found for {year} {series}"
    
    # Special case for 2013 series 2 drivers
    if year == 2013 and series == 2 and championship_type == 'Drivers\'':
        # Find the 4th wikitable after the heading
        current = heading
        for _ in range(4):
            current = current.find_next("table", {"class": "wikitable"})
            if not current:
                break
        table = current
    else:
        table = heading.find_next("table", {"class": "wikitable"})
    
    if not table:
        return None, f"No {championship_type} table found for {year} {series}"
    
    return table, None


def has_number_column(race_headers, year):
    """Check if the table has a number column."""
    if len(race_headers) > 2:
        second_header = remove_superscripts(race_headers[2])
        return "No." in second_header or ("No" == second_header and year == 2010)
    return False


def build_headers(race_headers, round_headers, year, series, file_suffix):
    """Build the complete header row for the CSV."""
    base_headers = ["Pos", "Team" if "team" in file_suffix.lower() else "Driver"]
    has_no_col = has_number_column(race_headers, year)
    col_index = 3 if has_no_col else 2
    
    combined_headers = base_headers[:]
    
    for i, th in enumerate(race_headers[col_index:], col_index):
        race_name = remove_superscripts(th, False)
        if not race_name or race_name in ['Points', 'Pts']:
            break
        
        colspan = int(th.get('colspan', 1))
        
        # Get round names
        if (year > 2012 and series == 3) or (year > 2016 and series == 2):
            race_rounds = get_round_names(round_headers, col_index, i, colspan)
        else:
            race_rounds = [f"R{r+1}" for r in range(colspan)]
        
        race_rounds.sort(key=lambda x: int(x.replace('R', '')) if x.replace('R', '').isdigit() else 999)
        
        for round_name in race_rounds:
            combined_headers.append(f"{race_name} {round_name}")
        
        col_index += colspan
    
    combined_headers.append("Points")
    return combined_headers, has_no_col


def get_round_names(round_headers, col_index, current_i, colspan):
    """Extract round names for the current race."""
    race_rounds = []
    round_start_idx = current_i - col_index
    
    for j in range(colspan):
        round_idx = round_start_idx + j
        if round_idx < len(round_headers):
            round_name = remove_superscripts(round_headers[round_idx], False)
            if round_name:
                race_rounds.append(round_name)
    
    return race_rounds


def get_data_rows(all_rows, year, series, championship_type):
    """Extract and clean data rows from the table."""
    # Skip header rows
    if (year > 2012 and series == 3) or (year > 2016 and series == 2):
        data_rows = all_rows[2:]
    else:
        data_rows = all_rows[1:]
    
    # Remove footer rows
    footer_rows_to_remove = get_footer_rows_count(year, series, championship_type)
    if len(data_rows) > footer_rows_to_remove:
        data_rows = data_rows[:-footer_rows_to_remove]
    
    return data_rows


def get_footer_rows_count(year, series, championship_type):
    """Determine how many footer rows to remove."""
    if ((year < 2013 and series == 3) or (series == 2 and year < 2017) or series == 1):
        return 2
    elif championship_type == "Drivers'" and year == 2020 and series == 3:
        return 4
    return 3


def process_table_row(cells, combined_headers, has_no_col, rowspan_tracker):
    """Process a single table row and return the formatted data."""
    if len(cells) < 3:
        return None
    
    row_data = []
    cell_index = 0
    
    # Handle position with rowspan
    if rowspan_tracker['pos_rowspan'] <= 0:
        pos_cell = cells[cell_index]
        rowspan_tracker['current_pos'] = remove_superscripts(pos_cell)
        rowspan_tracker['pos_rowspan'] = int(pos_cell.get('rowspan', 1))
        cell_index += 1
    rowspan_tracker['pos_rowspan'] -= 1
    row_data.append(rowspan_tracker['current_pos'])
    
    # Handle team/driver with rowspan
    if rowspan_tracker['team_rowspan'] <= 0:
        team_cell = cells[cell_index]
        rowspan_tracker['current_team'] = remove_superscripts(team_cell)
        rowspan_tracker['team_rowspan'] = int(team_cell.get('rowspan', 1))
        cell_index += 1
    rowspan_tracker['team_rowspan'] -= 1
    row_data.append(rowspan_tracker['current_team'])
    
    # Skip No. column if present
    if has_no_col and cell_index < len(cells):
        cell_index += 1
    
    # Process race columns
    num_race_columns = len(combined_headers) - 3
    race_cells = cells[cell_index:cell_index + num_race_columns]
    
    for cell in race_cells:
        row_data.append(remove_superscripts(cell, False))
    
    # Pad race columns if needed
    while len(row_data) < len(combined_headers) - 1:
        row_data.append("")
    
    # Handle points with rowspan
    if rowspan_tracker['points_rowspan'] <= 0:
        if cell_index + num_race_columns < len(cells):
            points_cell = cells[cell_index + num_race_columns]
            rowspan_tracker['current_points'] = remove_superscripts(points_cell)
            rowspan_tracker['points_rowspan'] = int(points_cell.get('rowspan', 1))
        else:
            # If no points cell found, use empty string
            rowspan_tracker['current_points'] = ""
            rowspan_tracker['points_rowspan'] = 1
    rowspan_tracker['points_rowspan'] -= 1
    row_data.append(rowspan_tracker['current_points'])
    
    # Ensure correct column count
    row_data = row_data[:len(combined_headers)]
    while len(row_data) < len(combined_headers):
        row_data.append("")
    
    return row_data


def write_championship_csv(file_path, combined_headers, data_rows, has_no_col):
    """Write the processed championship data to CSV."""
    rowspan_tracker = {
        'pos_rowspan': 0,
        'team_rowspan': 0,
        'points_rowspan': 0,
        'current_pos': "",
        'current_team': "",
        'current_points': ""
    }
    
    with open(file_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(combined_headers)
        
        for row in data_rows:
            cells = row.find_all(["th", "td"])
            row_data = process_table_row(cells, combined_headers, has_no_col, rowspan_tracker)
            if row_data:
                writer.writerow(row_data)


def process_championship(soup, championship_type, year, file_suffix, series):
    """Main function to process a championship table and save as CSV."""
    table, error = find_championship_table(soup, championship_type, series, year)
    if error:
        print(error)
        return
    
    all_rows = table.find_all("tr")
    if len(all_rows) < 3:
        print(f"Not enough rows in {championship_type} {year} {series}")
        return
    
    # Extract headers
    race_headers = all_rows[0].find_all("th")
    round_headers = None
    if (year > 2012 and series == 3) or (year > 2016 and series == 2):
        round_headers = all_rows[1].find_all("th")
    
    combined_headers, has_no_col = build_headers(race_headers, round_headers, year, series, file_suffix)
    
    # Get data rows
    data_rows = get_data_rows(all_rows, year, series, championship_type)
    
    # Create output file
    dir_path = os.path.join(DATA_DIR, f"F{series}", str(year))
    filename = f"f{series}_{year}_{file_suffix}.csv"
    os.makedirs(dir_path, exist_ok=True)
    full_path = os.path.join(dir_path, filename)
    
    # Write CSV
    write_championship_csv(full_path, combined_headers, data_rows, has_no_col)