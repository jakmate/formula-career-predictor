import pytest
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup
from app.scrapers.qualifying_scraper import (
    add_time_gap,
    extract_race_report_links,
    process_qualifying_data,
    parse_time_to_seconds,
    normalize_time_str,
    process_two_table_qualifying,
    extract_quali_table_data,
    save_qualifying_data,
    scrape_quali
)


class TestAddTimeGap:
    def test_add_gap_to_time_with_minutes(self):
        assert add_time_gap("1:19.429", "0.016") == "1:19.445"

    def test_add_gap_to_time_without_minutes(self):
        assert add_time_gap("59.429", "0.500") == "59.929"

    def test_add_gap_causing_minute_overflow(self):
        assert add_time_gap("1:59.900", "0.200") == "2:00.100"

    def test_invalid_time_returns_original(self):
        assert add_time_gap("invalid", "0.016") == "invalid"


class TestParseTimeToSeconds:
    def test_parse_time_with_colon(self):
        assert parse_time_to_seconds("1:19.429") == 79.429

    def test_parse_time_with_dots(self):
        assert parse_time_to_seconds("1.19.429") == 79.429

    def test_invalid_format_raises_error(self):
        with pytest.raises(ValueError):
            parse_time_to_seconds("59.123")


class TestNormalizeTimeStr:
    def test_normalize_dotted_time(self):
        assert normalize_time_str("1.19.429") == "1:19.429"

    def test_already_normalized_time(self):
        assert normalize_time_str("1:19.429") == "1:19.429"


class TestExtractRaceReportLinks:
    def test_extract_links_from_season_summary(self):
        html = """
        <h3 id="Season_summary">Season summary</h3>
        <table class="wikitable">
            <tr><th>Round</th></tr>
            <tr><td><a href="/wiki/2024_Race_1">Report</a></td></tr>
            <tr><td><a href="/wiki/2024_Race_2">Report</a></td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "lxml")
        links = extract_race_report_links(soup)
        assert len(links) == 2
        assert links[0] == "https://en.wikipedia.org/wiki/2024_Race_1"
        assert links[1] == "https://en.wikipedia.org/wiki/2024_Race_2"

    def test_no_season_summary_returns_empty(self):
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "lxml")
        links = extract_race_report_links(soup)
        assert links == []

    def test_alternative_heading_ids(self):
        html = """
        <h2 id="Results">Results</h2>
        <table class="wikitable">
            <tr><th>Round</th></tr>
            <tr><td><a href="/wiki/2024_Race_1">Report</a></td></tr>
        </table>
        """
        soup = BeautifulSoup(html, "lxml")
        links = extract_race_report_links(soup)
        assert len(links) == 1


class TestExtractQualiTableData:
    def test_extract_single_row_header(self):
        html = """
        <table class="wikitable">
            <tr>
                <th>Pos</th><th>No</th><th>Name</th>
                <th>Constructor</th><th>Time</th><th>Grid</th>
            </tr>
            <tr>
                <td>1</td><td>1</td><td>Max Verstappen</td>
                <td>Red Bull</td><td>1:19.429</td><td>1</td>
            </tr>
        </table>
        """
        table = BeautifulSoup(html, "lxml").find("table")
        result = extract_quali_table_data(table)

        assert result is not None
        assert result['headers'] == ['Pos.', 'No.', 'Driver', 'Team', 'Time', 'Grid']
        assert len(result['data']) == 1
        assert result['data'][0] == ['1', '1', 'Max Verstappen', 'Red Bull', '1:19.429', '1']

    def test_extract_two_row_header(self):
        html = """
        <table class="wikitable">
            <tr>
                <th rowspan="2">Pos</th>
                <th rowspan="2">No</th>
                <th rowspan="2">Name</th>
                <th rowspan="2">Constructor</th>
                <th colspan="3">Qualifying</th>
                <th rowspan="2">Grid</th>
            </tr>
            <tr>
                <th>Part 1</th><th>Part 2</th><th>Part 3</th>
            </tr>
            <tr>
                <td>1</td><td>1</td><td>Max Verstappen</td><td>Red Bull</td>
                <td>1:20.1</td><td>1:19.5</td><td>1:19.0</td><td>1</td>
            </tr>
        </table>
        """
        table = BeautifulSoup(html, "lxml").find("table")
        result = extract_quali_table_data(table)

        assert result is not None
        assert result['headers'] == ['Pos.', 'No.', 'Driver', 'Team', 'Q1', 'Q2', 'Q3', 'Grid']
        assert len(result['data']) == 1

    def test_convert_time_gaps_to_actual_times(self):
        html = """
        <table class="wikitable">
            <tr>
                <th>Pos</th><th>No</th><th>Name</th><th>Constructor</th><th>Time/Gap</th><th>Grid</th>
            </tr>
            <tr>
                <td>1</td><td>1</td><td>Driver 1</td><td>Team 1</td><td>1:19.429</td><td>1</td>
            </tr>
            <tr>
                <td>2</td><td>2</td><td>Driver 2</td><td>Team 2</td><td>+0.016</td><td>2</td>
            </tr>
        </table>
        """
        table = BeautifulSoup(html, "lxml").find("table")
        result = extract_quali_table_data(table)

        assert result['headers'][4] == 'Time'  # Renamed from Time/Gap
        assert result['data'][0][4] == '1:19.429'
        assert result['data'][1][4] == '1:19.445'

    def test_normalize_dotted_times(self):
        html = """
        <table class="wikitable">
            <tr>
                <th>Pos</th><th>No</th><th>Name</th><th>Constructor</th><th>Time</th><th>Grid</th>
            </tr>
            <tr>
                <td>1</td><td>1</td><td>Driver 1</td><td>Team 1</td><td>1.19.429</td><td>1</td>
            </tr>
        </table>
        """
        table = BeautifulSoup(html, "lxml").find("table")
        result = extract_quali_table_data(table)

        assert result['data'][0][4] == '1:19.429'

    def test_truncate_long_grid_numbers(self):
        html = """
        <table class="wikitable">
            <tr>
                <th>Pos</th><th>No</th><th>Name</th><th>Constructor</th><th>Time</th><th>Grid</th>
            </tr>
            <tr>
                <td>1</td><td>1</td><td>Driver</td><td>Team</td><td>1:19.429</td><td>123</td>
            </tr>
        </table>
        """
        table = BeautifulSoup(html, "lxml").find("table")
        result = extract_quali_table_data(table)

        assert result['data'][0][5] == '12'

    def test_insufficient_rows_returns_none(self):
        html = """
        <table class="wikitable">
            <tr><th>Header</th></tr>
        </table>
        """
        table = BeautifulSoup(html, "lxml").find("table")
        result = extract_quali_table_data(table)

        assert result is None


class TestProcessTwoTableQualifying:
    @patch('app.scrapers.qualifying_scraper.extract_quali_table_data')
    def test_alternating_faster_grid(self, mock_extract):
        mock_extract.side_effect = [
            {
                'headers': ['Pos.', 'No.', 'Driver', 'Team', 'Time', 'Grid'],
                'data': [
                    ['1', '1', 'Driver A1', 'Team A', '1:19.000', '1'],
                    ['2', '2', 'Driver A2', 'Team A', '1:19.500', '2']
                ]
            },
            {
                'headers': ['Pos.', 'No.', 'Driver', 'Team', 'Time', 'Grid'],
                'data': [
                    ['1', '3', 'Driver B1', 'Team B', '1:19.200', '1'],
                    ['2', '4', 'Driver B2', 'Team B', '1:19.600', '2']
                ]
            }
        ]

        group_a_head = Mock()
        group_b_head = Mock()
        group_a_head.find_next.return_value = Mock()
        group_b_head.find_next.return_value = Mock()

        result = process_two_table_qualifying(group_a_head, group_b_head, "Round 1", "url")

        assert result is not None
        assert len(result['data']) == 4
        # Check alternating pattern: A1, B1, A2, B2
        assert result['data'][0][2] == 'Driver A1'
        assert result['data'][1][2] == 'Driver B1'
        assert result['data'][2][2] == 'Driver A2'
        assert result['data'][3][2] == 'Driver B2'
        # Check position renumbering
        assert result['data'][0][0] == '1'
        assert result['data'][1][0] == '2'
        assert result['data'][2][0] == '3'
        assert result['data'][3][0] == '4'


class TestProcessQualifyingData:
    @patch('app.scrapers.qualifying_scraper.safe_request')
    def test_process_standard_qualifying(self, mock_request):
        html = """
        <h3 id="Qualifying">Qualifying</h3>
        <table class="wikitable">
            <tr>
                <th>Pos</th><th>No</th><th>Name</th><th>Constructor</th><th>Time</th><th>Grid</th>
            </tr>
            <tr>
                <td>1</td><td>1</td><td>Driver</td><td>Team</td><td>1:19.429</td><td>1</td>
            </tr>
        </table>
        """
        mock_response = Mock()
        mock_response.text = html
        mock_request.return_value = mock_response

        session = Mock()
        result = process_qualifying_data("http://test.com", "Round 1", session)

        assert result is not None
        assert result['round_info'] == "Round 1"
        assert result['url'] == "http://test.com"
        assert len(result['data']) == 1

    @patch('app.scrapers.qualifying_scraper.safe_request')
    def test_no_qualifying_section_returns_none(self, mock_request):
        html = "<html><body></body></html>"
        mock_response = Mock()
        mock_response.text = html
        mock_request.return_value = mock_response

        session = Mock()
        result = process_qualifying_data("http://test.com", "Round 1", session)

        assert result is None

    @patch('app.scrapers.qualifying_scraper.safe_request')
    def test_failed_request_returns_none(self, mock_request):
        mock_request.return_value = None

        session = Mock()
        result = process_qualifying_data("http://test.com", "Round 1", session)

        assert result is None


class TestSaveQualifyingData:
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    def test_save_qualifying_data(self, mock_makedirs, mock_open):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        qualifying_results = [
            {
                'headers': ['Pos.', 'No.', 'Driver', 'Team', 'Time', 'Grid'],
                'data': [['1', '1', 'Driver', 'Team', '1:19.429', '1']],
                'round_info': 'Round 1',
                'url': 'http://test.com'
            }
        ]

        with patch('app.scrapers.qualifying_scraper.DATA_DIR', '/data'):
            save_qualifying_data(qualifying_results, 2024, 1)

        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()

    @patch('os.makedirs')
    def test_save_skips_none_results(self, mock_makedirs):
        qualifying_results = [None, None]

        with patch('app.scrapers.qualifying_scraper.DATA_DIR', '/data'):
            save_qualifying_data(qualifying_results, 2024, 1)

        mock_makedirs.assert_called_once()


class TestScrapeQuali:
    @patch('app.scrapers.qualifying_scraper.save_qualifying_data')
    @patch('app.scrapers.qualifying_scraper.process_qualifying_data')
    @patch('app.scrapers.qualifying_scraper.extract_race_report_links')
    @patch('app.scrapers.qualifying_scraper.create_session')
    def test_scrape_quali_full_flow(self, mock_session, mock_extract, mock_process, mock_save):
        mock_extract.return_value = ['http://race1.com', 'http://race2.com']
        mock_process.side_effect = [
            {'headers': [], 'data': [], 'round_info': 'Round 1', 'url': 'http://race1.com'},
            {'headers': [], 'data': [], 'round_info': 'Round 2', 'url': 'http://race2.com'}
        ]

        soup = Mock()
        scrape_quali(soup, 2024, 1)

        assert mock_process.call_count == 2
        mock_save.assert_called_once()

    @patch('app.scrapers.qualifying_scraper.extract_race_report_links')
    @patch('app.scrapers.qualifying_scraper.create_session')
    def test_scrape_quali_no_links(self, mock_session, mock_extract):
        mock_extract.return_value = []

        soup = Mock()
        scrape_quali(soup, 2024, 1)

        # Should return early without processing
        mock_extract.assert_called_once()
