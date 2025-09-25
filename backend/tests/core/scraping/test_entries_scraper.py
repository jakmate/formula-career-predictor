from unittest.mock import Mock, patch, mock_open
from bs4 import BeautifulSoup

from app.core.scraping.entries_scraper import (
    get_entries_heading_id,
    find_entries_table,
    get_rowspan_column_count,
    should_remove_footer_row,
    process_headers,
    process_multirow_headers,
    process_single_row_headers,
    clean_headers,
    remove_footer_if_needed,
    process_rowspan_columns,
    process_f1_modern_drivers,
    write_f1_modern_rows,
    process_standard_row,
    process_entries
)


class TestGetEntriesHeadingId:
    def test_f1_2018_plus(self):
        assert get_entries_heading_id(2020, 1) == "Entries"
        assert get_entries_heading_id(2018, 1) == "Entries"

    def test_f1_2016(self):
        assert get_entries_heading_id(2016, 1) == "Entries"

    def test_f1_older(self):
        assert get_entries_heading_id(2015, 1) == "Teams_and_drivers"
        assert get_entries_heading_id(2010, 1) == "Teams_and_drivers"

    def test_f2_2018(self):
        assert get_entries_heading_id(2018, 2) == "Entries"

    def test_f2_older(self):
        assert get_entries_heading_id(2017, 2) == "Teams_and_drivers"

    def test_f3_2019(self):
        assert get_entries_heading_id(2019, 3) == "Teams_and_drivers"

    def test_f3_newer(self):
        assert get_entries_heading_id(2020, 3) == "Entries"


class TestFindEntriesTable:
    def test_finds_table_successfully(self):
        html = '''
        <h2 id="Entries">Entries</h2>
        <table class="wikitable">
            <tr><th>Team</th></tr>
        </table>
        '''
        soup = BeautifulSoup(html, 'lxml')
        table = find_entries_table(soup, 2020, 1)
        assert table is not None
        assert table.name == "table"

    def test_no_heading_found(self):
        html = '<div>No heading</div>'
        soup = BeautifulSoup(html, 'lxml')
        table = find_entries_table(soup, 2020, 1)
        assert table is None

    def test_no_table_after_heading(self):
        html = '<h2 id="Entries">Entries</h2><p>No table</p>'
        soup = BeautifulSoup(html, 'lxml')
        table = find_entries_table(soup, 2020, 1)
        assert table is None


class TestGetRowspanColumnCount:
    def test_f1_old(self):
        assert get_rowspan_column_count(1, 2013) == 6
        assert get_rowspan_column_count(1, 2010) == 6

    def test_f1_new(self):
        assert get_rowspan_column_count(1, 2014) == 4
        assert get_rowspan_column_count(1, 2020) == 4

    def test_other_series(self):
        assert get_rowspan_column_count(2, 2020) == 2
        assert get_rowspan_column_count(3, 2019) == 2


class TestShouldRemoveFooterRow:
    def test_f2_2017(self):
        assert not should_remove_footer_row(2, 2017)

    def test_f3_old(self):
        assert not should_remove_footer_row(3, 2016)
        assert not should_remove_footer_row(3, 2015)

    def test_f1_old(self):
        assert not should_remove_footer_row(1, 2013)
        assert not should_remove_footer_row(1, 2010)

    def test_should_remove(self):
        assert should_remove_footer_row(1, 2020)
        assert should_remove_footer_row(2, 2020)
        assert should_remove_footer_row(3, 2020)


class TestProcessHeaders:
    def test_multirow_headers_f1_2016_plus(self):
        html = '''
        <tr><th colspan="2">Team</th><th>Driver</th></tr>
        <tr><th>Name</th><th>Constructor</th><th>Name</th></tr>
        <tr><td>Data</td></tr>
        '''
        soup = BeautifulSoup(html, 'lxml')
        rows = soup.find_all('tr')

        with patch('app.core.scraping.entries_scraper.process_multirow_headers') as mock_multi:
            mock_multi.return_value = (['Team', 'Constructor', 'Driver'], rows[2:])
            headers, data_rows = process_headers(rows, 1, 2016)
            mock_multi.assert_called_once_with(rows)

    def test_single_row_headers(self):
        html = '''
        <tr><th>Team</th><th>Driver</th></tr>
        <tr><td>Data</td></tr>
        '''
        soup = BeautifulSoup(html, 'lxml')
        rows = soup.find_all('tr')

        with patch('app.core.scraping.entries_scraper.process_single_row_headers') as mock_single:
            mock_single.return_value = (['Team', 'Driver'], rows[1:])
            headers, data_rows = process_headers(rows, 1, 2015)
            mock_single.assert_called_once_with(rows)


class TestProcessMultirowHeaders:
    def test_colspan_expansion(self):
        html = '''
        <tr><th colspan="2">Team<sup>[1]</sup></th><th>Driver</th></tr>
        <tr><th>Name</th><th>Constructor</th><th>Name</th></tr>
        <tr><td>Data</td></tr>
        '''
        soup = BeautifulSoup(html, 'lxml')
        rows = soup.find_all('tr')

        with patch('app.core.scraping.scraping_utils.remove_superscripts') as mock_remove:
            mock_remove.side_effect = ['Team', 'Name', 'Constructor', 'Name']
            headers, data_rows = process_multirow_headers(rows)

            assert headers == ['Name', 'Constructor', 'Driver']
            assert len(data_rows) == 1


class TestProcessSingleRowHeaders:
    def test_basic_headers(self):
        html = '''
        <tr><th>Team<sup>[1]</sup></th><th>Driver</th></tr>
        <tr><td>Data</td></tr>
        '''
        soup = BeautifulSoup(html, 'lxml')
        rows = soup.find_all('tr')

        with patch('app.core.scraping.scraping_utils.remove_superscripts') as mock_remove:
            mock_remove.side_effect = ['Team', 'Driver']
            headers, data_rows = process_single_row_headers(rows)

            assert headers == ['Team', 'Driver']
            assert len(data_rows) == 1


class TestCleanHeaders:
    def test_header_mapping(self):
        headers = ['Entrant', 'Drivers', 'Engine', 'Points']
        clean, unwanted = clean_headers(headers)

        assert clean == ['Team', 'Driver', 'Points']
        assert 2 in unwanted  # Engine column index

    def test_unwanted_columns(self):
        headers = ['Team', 'Driver', 'Chassis', 'Status']
        clean, unwanted = clean_headers(headers)

        assert clean == ['Team', 'Driver']
        assert unwanted == [2, 3]


class TestRemoveFooterIfNeeded:
    def test_should_not_remove(self):
        mock_rows = [Mock(), Mock()]
        result = remove_footer_if_needed(mock_rows, 5, 2, 2017)
        assert result == mock_rows

    def test_remove_short_footer(self):
        html = '''
        <tr><td>A</td><td>B</td><td>C</td></tr>
        <tr><td>Footer</td></tr>
        '''
        soup = BeautifulSoup(html, 'lxml')
        rows = soup.find_all('tr')

        result = remove_footer_if_needed(rows, 3, 1, 2020)
        assert len(result) == 1

    def test_keep_full_footer(self):
        html = '''
        <tr><td>A</td><td>B</td><td>C</td></tr>
        <tr><td>D</td><td>E</td><td>F</td></tr>
        '''
        soup = BeautifulSoup(html, 'lxml')
        rows = soup.find_all('tr')

        result = remove_footer_if_needed(rows, 3, 1, 2020)
        assert len(result) == 2


class TestProcessRowspanColumns:
    def test_basic_rowspan(self):
        html = '<tr><td rowspan="2">Team A</td><td>Driver 1</td></tr>'
        soup = BeautifulSoup(html, 'lxml')
        cells = soup.find('tr').find_all(['td', 'th'])

        trackers = [{'value': '', 'remaining': 0}, {'value': '', 'remaining': 0}]

        with patch('app.core.scraping.scraping_utils.remove_superscripts') as mock_remove:
            mock_remove.side_effect = ['Team A', 'Driver 1']
            row_data, cell_index = process_rowspan_columns(trackers, cells, 2)

            assert row_data == ['Team A', 'Driver 1']
            assert trackers[0]['remaining'] == 1
            assert cell_index == 2

    def test_reuse_rowspan_value(self):
        # Create a proper HTML cell instead of Mock
        html = '<td>Driver 2</td>'
        soup = BeautifulSoup(html, 'lxml')
        cells = [soup.find('td')]

        trackers = [
            {'value': 'Team A', 'remaining': 1},
            {'value': '', 'remaining': 0}
        ]

        with patch('app.core.scraping.scraping_utils.remove_superscripts') as mock_remove:
            mock_remove.return_value = 'Driver 2'
            row_data, cell_index = process_rowspan_columns(trackers, cells, 2)

            assert row_data == ['Team A', 'Driver 2']
            assert trackers[0]['remaining'] == 0


class TestProcessF1ModernDrivers:
    def test_merge_numbered_lines(self):
        html = '''
        <td>
            Lewis Hamilton<br/>
            44<br/>
            –<br/>
            Valtteri Bottas<br/>
            77
        </td>
        '''
        soup = BeautifulSoup(html, 'lxml')
        cells = [soup.find('td')]

        result = process_f1_modern_drivers(cells)
        expected = [['Lewis Hamilton', '44–', 'Valtteri Bottas', '77']]
        assert result == expected


class TestWriteF1ModernRows:
    def test_write_multiple_drivers(self):
        writer = Mock()
        row_data = ['Team A', 'Constructor', 'Engine', 'Car']
        processed_cells = [['Driver 1', 'Driver 2'], ['44', '77']]
        unwanted_indices = []

        write_f1_modern_rows(writer, row_data, processed_cells, unwanted_indices)

        assert writer.writerow.call_count == 2
        writer.writerow.assert_any_call(['Team A', 'Constructor', 'Engine',
                                         'Car', 'Driver 1', '44'])
        writer.writerow.assert_any_call(['Team A', 'Constructor', 'Engine',
                                         'Car', 'Driver 2', '77'])


class TestProcessStandardRow:
    def test_robert_visoiu_fix(self):
        # Create proper HTML cells
        html = '<tr><td>Team</td><td>Robert Visoiu</td></tr>'
        soup = BeautifulSoup(html, 'lxml')
        cells = soup.find('tr').find_all(['td', 'th'])

        with patch('app.core.scraping.scraping_utils.remove_superscripts') as mock_remove:
            mock_remove.side_effect = ['Team', 'Robert Visoiu']

            result = process_standard_row(cells, 0, [], 2, 1, 3, [])
            assert result[1] == 'Robert Vișoiu'


@patch('app.core.scraping.scraping_utils.create_output_file')
@patch('builtins.open', new_callable=mock_open)
class TestProcessEntries:
    def test_no_table_found(self, mock_file, mock_create):
        soup = BeautifulSoup('<div>No table</div>', 'lxml')

        with patch('app.core.scraping.entries_scraper.find_entries_table') as mock_find:
            mock_find.return_value = None
            process_entries(soup, 2020, 1)
            mock_file.assert_not_called()

    def test_insufficient_rows(self, mock_file, mock_create):
        html = '<table class="wikitable"><tr><th>Header</th></tr></table>'
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')

        with patch('app.core.scraping.entries_scraper.find_entries_table') as mock_find:
            mock_find.return_value = table
            process_entries(soup, 2020, 1)
            mock_file.assert_not_called()

    def test_successful_processing(self, mock_file, mock_create):
        html = '''
        <table class="wikitable">
            <tr><th>Team</th><th>Driver</th></tr>
            <tr><td>Mercedes</td><td>Hamilton</td></tr>
            <tr><td>Red Bull</td><td>Verstappen</td></tr>
        </table>
        '''
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')

        mock_create.return_value = '/path/to/file.csv'

        with patch('app.core.scraping.entries_scraper.find_entries_table') as mock_find:
            mock_find.return_value = table

            with patch('csv.writer') as mock_writer_class:
                mock_writer = Mock()
                mock_writer_class.return_value = mock_writer

                process_entries(soup, 2020, 2)

                mock_file.assert_called_once()
                mock_writer.writerow.assert_called()  # At least header written
