import csv
import os
from bs4 import BeautifulSoup, SoupStrainer
from app.config import DATA_DIR
from app.core.scraping.scraping_utils import create_session, remove_superscripts, safe_request

COLUMN_MAPPING = {
    'Name': 'Driver',
    'Entrant': 'Team',
    'Part 1': 'Q1',
    'Part 2': 'Q2',
    'Part 3': 'Q3',
    'Finalgrid': 'Grid',
    'Pos': 'Pos.',
    'Carno.': 'No.',
    'No': 'No.',
    'Constructor': 'Team',
    'GridFR': 'Grid',
    'R2': 'Grid',
    'Q2 Time': 'Time'
}


def add_time_gap(base_time, gap):
    """Add gap time to base time (e.g., '1:19.429' + '0.016' = '1:19.445')"""
    try:
        # Parse base time
        if ':' in base_time:
            time_parts = base_time.split(':')
            minutes = int(time_parts[0])
            seconds = float(time_parts[1])
        else:
            minutes = 0
            seconds = float(base_time)

        # Add gap
        total_seconds = seconds + float(gap)

        # Minute overflow
        if total_seconds >= 60:
            minutes += int(total_seconds // 60)
            total_seconds = total_seconds % 60

        # Format result
        if minutes > 0:
            return f"{minutes}:{total_seconds:06.3f}"
        else:
            return f"{total_seconds:.3f}"
    except (ValueError, IndexError):
        # If parsing fails, return original base time
        return base_time


def extract_race_report_links(soup):
    """Extract race report links from the season summary table"""
    # Find Season summary heading
    season_heading = (soup.find("h3", {"id": "Season_summary"}) or
                      soup.find("h3", {"id": "Summary"}) or
                      soup.find("h2", {"id": "Results"}) or
                      soup.find("h2", {"id": "Results_and_standings"}))

    if not season_heading:
        print("No season summary table found")
        return []

    table = season_heading.find_next("table", {"class": "wikitable"})
    if not table:
        print("No season summary table found")
        return []

    race_links = []
    for row in table.find_all("tr")[1:]:
        report_link = row.find("a", string="Report")
        if report_link:
            href = report_link.get("href")
            if href and href.startswith("/wiki/"):
                race_links.append("https://en.wikipedia.org" + href)

    return race_links


def process_qualifying_data(race_url, round_info, session):
    """Process qualifying data from a race report page"""
    try:
        response = safe_request(session, race_url)
        if response is None:
            print(f"Failed to fetch {race_url}")
            return None

        parse_only = SoupStrainer(["h2", "h3", "h4", "dt", "table"])
        soup = BeautifulSoup(response.text, "lxml", parse_only=parse_only)

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
            result = process_two_table_qualifying(group_a_head, group_b_head, round_info, race_url)
        else:
            result = process_single_qualifying_table(qualifying_heading, round_info, race_url)

        return result

    except Exception as e:
        print(f"Error processing qualifying data from {race_url}: {str(e)}")
        return None


def parse_time_to_seconds(t):
    """Convert a time string to total seconds."""
    t = t.strip()
    # Case 1: "M:SS.mmm"
    if ':' in t:
        minutes, sec_ms = t.split(':', 1)
    # Case 2: "M.SS.mmm"
    elif t.count('.') >= 2:
        minutes, sec_ms = t.split('.', 1)
    else:
        raise ValueError(f"Unrecognized time format: '{t}'")
    return int(minutes) * 60 + float(sec_ms)


def normalize_time_str(t):
    """Turn 'M.SS.mmm' into 'M:SS.mmm'."""
    t = t.strip()
    if t.count('.') >= 2:
        mins, rest = t.split('.', 1)
        return f"{mins}:{rest}"
    return t


def process_two_table_qualifying(group_a_head, group_b_head, round_info, race_url):
    """Process Monte Carlo qualifying with Group A and Group B"""
    try:
        # Process Group A
        group_a_table = group_a_head.find_next("table", {"class": "wikitable"})
        group_a_data = extract_quali_table_data(group_a_table) if group_a_table else None

        # Process Group B
        group_b_table = group_b_head.find_next("table", {"class": "wikitable"})
        group_b_data = extract_quali_table_data(group_b_table) if group_b_table else None

        if not group_a_data and not group_b_data:
            print(f"No qualifying data found in either group for {race_url}")
            return None

        # Get headers from the first available table
        headers = group_a_data['headers'] if group_a_data else group_b_data['headers']

        group_a_rows = group_a_data['data'] if group_a_data else []
        group_b_rows = group_b_data['data'] if group_b_data else []

        # Get faster group
        time_idx = headers.index("Time")
        tA = parse_time_to_seconds(group_a_rows[0][time_idx])
        tB = parse_time_to_seconds(group_b_rows[0][time_idx])
        start_with_a = (tA <= tB)

        # Create alternating grid pattern
        combined_data = []
        max_len = max(len(group_a_rows), len(group_b_rows))

        # Determine order: ['A','B'] or ['B','A']
        order = ['A', 'B'] if start_with_a else ['B', 'A']
        pos = 1

        for i in range(max_len):
            for grp in order:
                rows = group_a_rows if grp == 'A' else group_b_rows
                if i < len(rows):
                    row = rows[i]
                    row[0] = str(pos)  # update Pos.
                    combined_data.append(row)
                    pos += 1

        return {
            'headers': headers,
            'data': combined_data,
            'round_info': round_info,
            'url': race_url,
        }

    except Exception as e:
        print(f"Error processing Monte Carlo qualifying from {race_url}: {str(e)}")
        return None


def process_single_qualifying_table(qualifying_heading, round_info, race_url):
    """Process standard single qualifying table, normalizing any dotted time formats."""
    try:
        table = qualifying_heading.find_next("table", {"class": "wikitable"})
        if not table:
            print(f"No qualifying table found for {race_url}")
            return None

        table_data = extract_quali_table_data(table)
        if not table_data:
            return None

        return {
            'headers': table_data['headers'],
            'data': table_data['data'],
            'round_info': round_info,
            'url': race_url,
        }

    except Exception as e:
        print(f"Error processing single qualifying table: {race_url}: {e}")
        return None


def extract_quali_table_data(table):
    """Extract data from a qualifying table"""
    try:
        all_rows = table.find_all("tr")
        if len(all_rows) < 2:
            return None

        # Check if we have a two-row header (F1 2015+)
        first_row = all_rows[0]
        second_row = all_rows[1] if len(all_rows) > 1 else None

        # Check if second row contains th elements
        has_two_row_header = (second_row and
                              second_row.find_all("th") and
                              not second_row.find_all("td"))

        if has_two_row_header:
            # Process two-row header structure
            headers = []
            first_row_headers = first_row.find_all("th")
            second_row_headers = second_row.find_all("th")

            # Collect rowspan=2 headers except the last one
            rowspan2_headers = []
            for i, th in enumerate(first_row_headers):
                rowspan = int(th.get("rowspan", 1))
                if rowspan == 2:
                    text = remove_superscripts(th, False)
                    if i < len(first_row_headers) - 1:
                        headers.append(text)
                    else:
                        rowspan2_headers.append(text)  # Save last one for end

            # Add the second row headers (Q1, Q2, Q3)
            headers.extend(remove_superscripts(th, False) for th in second_row_headers)

            # Add the last rowspan=2 header (Grid)
            headers.extend(rowspan2_headers)

            data_start_index = 2
        else:
            # Single row header
            headers = [remove_superscripts(th, False) for th in first_row.find_all("th")]
            data_start_index = 1  # Data starts from second row

        # Apply column mapping
        headers = [COLUMN_MAPPING.get(h, h) for h in headers]

        # Drop unwanted columns for single row headers
        if not has_two_row_header:
            columns_to_drop = {'R1', 'GridSR', 'Gap', 'Q1 Time', 'Rank'}
            indices_to_keep = [i for i, h in enumerate(headers) if h not in columns_to_drop]
            headers = [headers[i] for i in indices_to_keep]

        # Get data rows
        data_rows = []
        for row in all_rows[data_start_index:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 4:  # Need at least Pos, No, Driver, Team
                continue

            row_data = [remove_superscripts(cell, False) for cell in cells]

            if row_data:
                # Apply column filtering for single row headers
                if not has_two_row_header:
                    row_data = [row_data[i] for i in indices_to_keep if i < len(row_data)]

                # Process grid column (last column) truncation
                if row_data and row_data[-1].isdigit() and len(row_data[-1]) > 2:
                    row_data[-1] = row_data[-1][:2]  # Truncate to first two digits
                data_rows.append(row_data)

        # Convert Time/Gap column to actual times and normalise all times
        time_col_index = None
        for idx, header in enumerate(headers):
            if header.lower() == "time/gap":
                headers[idx] = "Time"  # Rename column
                time_col_index = idx
                break
            elif header == "Time":
                time_col_index = idx
                break

        if time_col_index is not None and data_rows:
            # Get pole position time (first row)
            pole_time = data_rows[0][time_col_index]

            # Convert gaps to actual times
            for row in data_rows:
                if time_col_index < len(row):
                    time_value = row[time_col_index]
                    if time_value.startswith('+') and pole_time:
                        time_value = add_time_gap(pole_time, time_value[1:])

                    if time_value.count('.') >= 2:
                        time_value = normalize_time_str(time_value)
                    row[time_col_index] = time_value

        return {'headers': headers, 'data': data_rows}

    except Exception as e:
        print(f"Error extracting table data: {str(e)}")
        return None


def save_qualifying_data(qualifying_results, year, series):
    """Save all qualifying data to CSV files"""
    if not qualifying_results:
        return

    dir_path = os.path.join(DATA_DIR, f"F{series}", str(year), "qualifying")
    base_filename = f"f{series}_{year}_qualifying"
    os.makedirs(dir_path, exist_ok=True)

    # Save each race's qualifying data separately
    for i, result in enumerate(qualifying_results, 1):
        if result is None:
            continue

        filename = f"{base_filename}_round_{i:02d}.csv"
        full_path = os.path.join(dir_path, filename)

        with open(full_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(result['headers'])
            writer.writerows(result['data'])


def scrape_quali(soup, year, num, session=None):
    if session is None:
        session = create_session()

    race_links = extract_race_report_links(soup)
    if not race_links:
        print(f"No race report links found for F{num} {year}")
        return

    quali_results = []
    for i, link in enumerate(race_links, 1):
        result = process_qualifying_data(link, f"Round {i}", session)
        quali_results.append(result)

    save_qualifying_data(quali_results, year, num)
