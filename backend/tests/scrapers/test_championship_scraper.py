from unittest.mock import patch, mock_open
from bs4 import BeautifulSoup

from app.scrapers.championship_scraper import (
    map_url,
    find_championship_table,
    build_headers,
    get_footer_rows_count,
    process_table_row,
    write_championship_csv,
    process_championship,
    has_number_column,
    get_round_names,
)


# Helper to create minimal soup
def make_soup(html):
    return BeautifulSoup(html, "lxml")


# Tests for map_url
def test_map_url_drivers_series1():
    assert map_url("Drivers'", 1, 2020) == "World_Drivers'_Championship_standings"


def test_map_url_teams_series1():
    assert map_url("Teams'", 1, 2020) == "World_Constructors'_Championship_standings"


def test_map_url_2013_series2_drivers():
    assert map_url("Drivers'", 2, 2013) == "Drivers'_championship"


def test_map_url_pre_2013():
    assert map_url("Drivers'", 2, 2012) == "Drivers'_Championship"


def test_map_url_2013_to_2022():
    assert map_url("Drivers'", 2, 2020) == "Drivers'_championship"


def test_map_url_2023_plus():
    assert map_url("Drivers'", 2, 2023) == "Drivers'_Championship_standings"


# find_championship_table tests
def test_find_championship_table_found():
    html = '<h3 id="World_Drivers\'_Championship_standings"></h3><table class="wikitable"></table>'
    soup = make_soup(html)
    table, error = find_championship_table(soup, "Drivers'", 1, 2020)
    assert table is not None
    assert error is None


def test_find_championship_table_not_found():
    soup = make_soup("<div></div>")
    table, error = find_championship_table(soup, "Drivers'", 1, 2020)
    assert table is None
    assert "No Drivers' heading found" in error


def test_find_championship_table_2013_series2_drivers():
    # Simulate 3 tables after heading
    html = """
    <h3 id="Drivers'_championship"></h3>
    <table class="wikitable"></table>
    <table class="wikitable"></table>
    <table class="wikitable" id="target"></table>
    """
    soup = make_soup(html)
    table, error = find_championship_table(soup, "Drivers'", 2, 2013)
    assert table is not None
    assert table.get("id") == "target"


def test_find_championship_table_2013_series2_not_enough_tables():
    html = '<h3 id="Drivers\'_championship"></h3><table class="wikitable"></table>'
    soup = make_soup(html)
    table, error = find_championship_table(soup, "Drivers'", 2, 2013)
    assert table is None
    assert "No Drivers' table found" in error


def test_has_number_column_true_for_no_and_year_2010():
    # third header text is "No" (without dot) and year==2010 should return True
    race_headers = [
        make_soup("<th>Pos</th>").find("th"),
        make_soup("<th>Driver</th>").find("th"),
        make_soup("<th>No</th>").find("th"),
    ]
    assert has_number_column(race_headers, 2010) is True


def test_has_number_column_false_when_too_short():
    # len <= 2 => False
    race_headers = [make_soup("<th>Pos</th>").find("th"), make_soup("<th>Driver</th>").find("th")]
    assert has_number_column(race_headers, 2010) is False


# build_headers tests (team label + Points break + round headers)
def test_build_headers_team_label():
    race_headers = [
        make_soup("<th>Pos</th>").find("th"),
        make_soup("<th>Team</th>").find("th"),
        make_soup("<th>Race1</th>").find("th")
    ]
    combined, _ = build_headers(race_headers, None, 2020, 1, "my_team_suffix")
    assert combined[1] == "Team"  # ensure 'Team' used instead of 'Driver'


def test_build_headers_points_break():
    # third header is "Points" so loop should break and only append final "Points"
    race_headers = [
        make_soup("<th>Pos</th>").find("th"),
        make_soup("<th>Driver</th>").find("th"),
        make_soup("<th>Points</th>").find("th"),
    ]
    combined, _ = build_headers(race_headers, None, 2020, 1, "drivers")
    assert combined == ["Pos", "Driver", "Points"]


def test_build_headers_with_round_headers():
    race_headers = [
        make_soup("<th>Pos</th>").find("th"),
        make_soup("<th>Driver</th>").find("th"),
        make_soup("<th>Race1</th>").find("th"),
    ]
    round_headers = [
        make_soup("<th>R2</th>").find("th"),  # non-sequential to test sorting
    ]
    combined, _ = build_headers(race_headers, round_headers, 2020, 3, "drivers")
    assert "Race1 R2" in combined


# get_round_names tests (colspan and out-of-range handling)
def test_get_round_names_partial_rounds():
    # round_headers shorter than colspan -> should only return available names
    round_headers = [
        make_soup("<th>R1</th>").find("th"),
    ]
    # col_index 2, current_i 2, colspan 2 means one valid, one out-of-range
    rounds = get_round_names(round_headers, 2, 2, 2)
    assert rounds == ["R1"]


# get_footer_rows_count tests
def test_get_footer_rows_count_series1():
    assert get_footer_rows_count(2020, 1, "Drivers'") == 2


def test_get_footer_rows_count_series2_pre_2017():
    assert get_footer_rows_count(2016, 2, "Drivers'") == 2


def test_get_footer_rows_count_series2_post_2016():
    assert get_footer_rows_count(2017, 2, "Drivers'") == 3


def test_get_footer_rows_count_series3_pre_2013():
    assert get_footer_rows_count(2012, 3, "Drivers'") == 2


def test_get_footer_rows_count_series3_post_2012():
    assert get_footer_rows_count(2013, 3, "Drivers'") == 3


def test_get_footer_rows_count_2020_series3_drivers():
    assert get_footer_rows_count(2020, 3, "Drivers'") == 4


# process_table_row tests (too few cells, rowspan, missing points, no. column)
def test_process_table_row_too_few_cells():
    cells = [make_soup("<td>only</td>").find("td"), make_soup("<td>two</td>").find("td")]
    headers = ["Pos", "Driver", "Points"]
    tracker = {'pos_rowspan': 0, 'team_rowspan': 0, 'points_rowspan': 0,
               'current_pos': '', 'current_team': '', 'current_points': ''}
    assert process_table_row(cells, headers, False, tracker) is None


def test_process_table_row_basic():
    cells = [
        make_soup("<td>1</td>").find("td"),
        make_soup("<td>Max</td>").find("td"),
        make_soup("<td>25</td>").find("td"),
    ]
    headers = ["Pos", "Driver", "Points"]
    tracker = {'pos_rowspan': 0, 'team_rowspan': 0, 'points_rowspan': 0,
               'current_pos': '', 'current_team': '', 'current_points': ''}
    result = process_table_row(cells, headers, False, tracker)
    assert result == ["1", "Max", "25"]


def test_process_table_row_rowspan_and_missing_points():
    # Simulate a row where pos and team are set with rowspan, but points cell is missing
    # Row has pos (rowspan=2), team (rowspan=2), race cell(s) but no points cell
    r1 = make_soup('<td rowspan="2">1</td>').find("td")
    r2 = make_soup('<td rowspan="2">Team A</td>').find("td")
    race_cell = make_soup('<td>R</td>').find("td")
    # only three cells in this row; points missing -> should go into missing-points branch
    cells = [r1, r2, race_cell]
    headers = ["Pos", "Team", "Race1", "Points"]  # combined headers length used by padding logic
    tracker = {'pos_rowspan': 0, 'team_rowspan': 0, 'points_rowspan': 0,
               'current_pos': '', 'current_team': '', 'current_points': ''}
    result = process_table_row(cells, headers, False, tracker)
    # Expect position and team to be taken, race cell value, and empty string for points
    assert result[0] == "1"
    assert result[1] == "Team A"
    assert "R" in result  # race value included
    assert result[-1] == ""  # points defaulted to empty string


def test_process_table_row_with_no_col_skips_column():
    # When has_no_col True, the third cell should be skipped (No. column)
    cells = [
        make_soup("<td>1</td>").find("td"),
        make_soup("<td>Driver</td>").find("td"),
        make_soup("<td>9</td>").find("td"),  # No. column that should be skipped
        make_soup("<td>R1</td>").find("td"),
        make_soup("<td>30</td>").find("td"),
    ]
    headers = ["Pos", "Driver", "R1", "Points"]
    tracker = {'pos_rowspan': 0, 'team_rowspan': 0, 'points_rowspan': 0,
               'current_pos': '', 'current_team': '', 'current_points': ''}
    result = process_table_row(cells, headers, True, tracker)
    # Should pick up Pos, Driver, then race R1, then Points
    assert result[0] == "1"
    assert result[1] == "Driver"
    assert "R1" in result
    assert result[-1] == "30"


# write_championship_csv tests (rowspan persistence across rows)
@patch("app.scrapers.championship_scraper.open", new_callable=mock_open)
def test_write_championship_csv_rowspan_across_rows(mock_file):
    headers = ["Pos", "Driver", "Race1", "Points"]

    # Row 1: provides pos/team/points with rowspan=2
    row1_html = """
    <tr>
      <td rowspan="2">1</td>
      <td rowspan="2">Driver A</td>
      <td>R1</td>
      <td rowspan="2">50</td>
    </tr>
    """
    # Row 2: only race cell (pos/team/points are covered by rowspan)
    row2_html = """
    <tr>
      <td>DNF</td>
    </tr>
    """

    row1 = make_soup(row1_html).find("tr")
    row2 = make_soup(row2_html).find("tr")
    data_rows = [row1, row2]

    write_championship_csv("dummy.csv", headers, data_rows, False)

    mock_file.assert_called_once_with("dummy.csv", "w", newline='', encoding="utf-8")
    handle = mock_file()
    # csv.writer writes lines - ensure header + two data rows exist
    handle.write.assert_any_call("Pos,Driver,Race1,Points\r\n")
    # Check first data row written
    # We assert at least one line contains Driver A
    written = "".join(call.args[0] for call in handle.write.mock_calls if call.args)
    assert "Driver A" in written
    assert "50" in written


# process_championship tests (full + error path)
@patch("app.scrapers.championship_scraper.write_championship_csv")
def test_process_championship_full(mock_write):
    html = """
    <h3 id="World_Drivers'_Championship_standings"></h3>
    <table class="wikitable">
        <tr><th>Pos</th><th>Driver</th><th>Race1</th><th>Points</th></tr>
        <tr><td>1</td><td>Max</td><td>25</td><td>25</td></tr>
        <tr><td>2</td><td>Lewis</td><td>18</td><td>18</td></tr>
        <tr><td></td><td></td><td></td><td></td></tr> <!-- footer -->
        <tr><td></td><td></td><td></td><td></td></tr>
    </table>
    """
    soup = make_soup(html)
    process_championship(soup, "Drivers'", 2020, "drivers", 1)

    assert mock_write.called
    args, _ = mock_write.call_args
    file_path, headers, data_rows, has_no_col = args
    assert "f1_2020_drivers.csv" in file_path
    assert headers == ["Pos", "Driver", "Race1 R1", "Points"]
    assert len(data_rows) == 2


@patch("builtins.print")
def test_process_championship_missing_heading_prints_error(mock_print):
    # no heading find_championship_table will return (None, error)
    soup = make_soup("<div></div>")
    process_championship(soup, "Drivers'", 2020, "drivers", 1)
    # error message contains this substring
    mock_print.assert_called()
    called_with = mock_print.call_args[0][0]
    assert "No Drivers' heading found for 2020 1" in called_with
