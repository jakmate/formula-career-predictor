import json
import pytest
import requests
from unittest.mock import Mock, patch, mock_open
import pandas as pd

from app.scrapers.driver_scraper import (
    get_driver_filename,
    search_wikidata_driver,
    extract_nationality_from_result,
    extract_dob_from_result,
    save_profile,
    scrape_driver_profile,
    get_all_drivers_from_data,
    scrape_drivers
)


class TestGetDriverFilename:
    def test_basic_name(self):
        assert get_driver_filename("Lewis Hamilton") == "lewis_hamilton.json"

    def test_multiple_spaces(self):
        assert get_driver_filename("Jean  Eric  Vergne") == "jean_eric_vergne.json"

    def test_hyphens(self):
        assert get_driver_filename("Jean-Eric Vergne") == "jean_eric_vergne.json"


class TestSearchWikidataDriver:
    def test_successful_search(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': {
                'bindings': [{
                    'person': {'value': 'http://www.wikidata.org/entity/Q1'},
                    'personLabel': {'value': 'Lewis Hamilton'},
                    'dob': {'value': '1985-01-07T00:00:00Z'},
                    'nationalityLabel': {'value': 'United Kingdom'}
                }]
            }
        }
        mock_session.get.return_value = mock_response

        result = search_wikidata_driver("Lewis Hamilton", mock_session)

        assert result is not None
        assert result['personLabel']['value'] == 'Lewis Hamilton'
        mock_session.get.assert_called_once()

    def test_driver_with_alias(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': {'bindings': [{'personLabel': {'value': 'Lucas Di Grassi'}}]}
        }
        mock_session.get.return_value = mock_response

        result = search_wikidata_driver("Lucas di Grassi", mock_session)

        assert result is not None
        # Check that the query used the alias
        call_args = mock_session.get.call_args
        assert 'Lucas Di Grassi' in call_args[1]['params']['query']

    def test_no_results(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'results': {'bindings': []}}
        mock_session.get.return_value = mock_response

        result = search_wikidata_driver("Unknown Driver", mock_session)

        assert result is None

    def test_request_exception(self):
        mock_session = Mock()
        mock_session.get.side_effect = requests.exceptions.Timeout()

        result = search_wikidata_driver("Lewis Hamilton", mock_session)

        assert result is None

    def test_non_200_status(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_session.get.return_value = mock_response

        result = search_wikidata_driver("Lewis Hamilton", mock_session)

        assert result is None


class TestExtractNationalityFromResult:
    def test_with_nationality_label(self):
        result = {'nationalityLabel': {'value': 'United Kingdom'}}
        assert extract_nationality_from_result(result) == 'United Kingdom'

    def test_with_citizenship_only(self):
        result = {
            'nationalityLabel': {'value': ''},
            'citizenshipLabel': {'value': 'France'}
        }
        assert extract_nationality_from_result(result) == 'France'

    def test_no_nationality_data(self):
        result = {}
        assert extract_nationality_from_result(result) is None


class TestExtractDobFromResult:
    def test_with_dob(self):
        result = {'dob': {'value': '1985-01-07T00:00:00Z'}}
        assert extract_dob_from_result(result) == '1985-01-07'

    def test_without_dob(self):
        result = {}
        assert extract_dob_from_result(result) is None


class TestSaveProfile:
    @patch('builtins.open', new_callable=mock_open)
    def test_save_profile(self, mock_file):
        profile = {
            'name': 'Lewis Hamilton',
            'dob': '1985-01-07',
            'nationality': 'United Kingdom'
        }

        save_profile('test.json', profile)

        mock_file.assert_called_once_with('test.json', 'w', encoding='utf-8')
        handle = mock_file()
        written_content = ''.join(call.args[0] for call in handle.write.call_args_list)
        parsed = json.loads(written_content)
        assert parsed == profile


class TestScrapeDriverProfile:
    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"name": "Lewis Hamilton", "scraped": true}') # noqa: 501
    def test_returns_cached_profile(self, mock_file, mock_exists):
        mock_exists.return_value = True
        mock_session = Mock()

        result = scrape_driver_profile("Lewis Hamilton", mock_session)

        assert result['name'] == 'Lewis Hamilton'
        assert result['scraped'] is True
        mock_session.get.assert_not_called()

    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.save_profile')
    @patch('app.scrapers.driver_scraper.search_wikidata_driver')
    def test_scrapes_new_profile(self, mock_search, mock_save, mock_exists):
        mock_exists.return_value = False
        mock_search.return_value = {
            'person': {'value': 'http://www.wikidata.org/entity/Q1'},
            'personLabel': {'value': 'Lewis Hamilton'},
            'dob': {'value': '1985-01-07T00:00:00Z'},
            'nationalityLabel': {'value': 'United Kingdom'}
        }
        mock_session = Mock()

        result = scrape_driver_profile("Lewis Hamilton", mock_session)

        assert result['name'] == 'Lewis Hamilton'
        assert result['dob'] == '1985-01-07'
        assert result['nationality'] == 'United Kingdom'
        assert result['wikidata_id'] == 'Q1'
        assert result['scraped'] is True
        mock_save.assert_called_once()

    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.save_profile')
    @patch('app.scrapers.driver_scraper.search_wikidata_driver')
    def test_handles_not_found(self, mock_search, mock_save, mock_exists):
        mock_exists.return_value = False
        mock_search.return_value = None
        mock_session = Mock()

        result = scrape_driver_profile("Unknown Driver", mock_session)

        assert result['name'] == 'Unknown Driver'
        assert result['dob'] is None
        assert result['nationality'] is None
        assert result['scraped'] is False
        mock_save.assert_called_once()

    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.save_profile')
    @patch('app.scrapers.driver_scraper.search_wikidata_driver')
    def test_handles_processing_error(self, mock_search, mock_save, mock_exists):
        mock_exists.return_value = False
        mock_search.return_value = {'person': {}}  # Missing required fields
        mock_session = Mock()

        result = scrape_driver_profile("Bad Data Driver", mock_session)

        assert result['name'] == 'Bad Data Driver'
        assert result['scraped'] is False
        assert 'error' in result


class TestGetAllDriversFromData:
    @patch('app.scrapers.driver_scraper.glob.glob')
    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.pd.read_csv')
    def test_extracts_drivers_from_files(self, mock_read_csv, mock_exists, mock_glob):
        mock_glob.return_value = ['data/F1/2023']
        mock_exists.return_value = True
        mock_df = pd.DataFrame({
            'Driver': ['Lewis Hamilton', 'Max Verstappen', 'Lewis Hamilton']
        })
        mock_read_csv.return_value = mock_df

        drivers = get_all_drivers_from_data()

        assert len(drivers) == 2
        assert 'Lewis Hamilton' in drivers
        assert 'Max Verstappen' in drivers

    @patch('app.scrapers.driver_scraper.glob.glob')
    def test_handles_no_data_dirs(self, mock_glob):
        mock_glob.return_value = []

        drivers = get_all_drivers_from_data()

        assert drivers == []

    @patch('app.scrapers.driver_scraper.glob.glob')
    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.pd.read_csv')
    def test_handles_missing_driver_column(self, mock_read_csv, mock_exists, mock_glob):
        mock_glob.return_value = ['data/F1/2023']
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'Team': ['Mercedes', 'Red Bull']})
        mock_read_csv.return_value = mock_df

        drivers = get_all_drivers_from_data()

        assert drivers == []

    @patch('app.scrapers.driver_scraper.glob.glob')
    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.pd.read_csv')
    def test_handles_read_error(self, mock_read_csv, mock_exists, mock_glob):
        mock_glob.return_value = ['data/F1/2023']
        mock_exists.return_value = True
        mock_read_csv.side_effect = Exception("Read error")

        drivers = get_all_drivers_from_data()

        assert drivers == []


class TestScrapeDrivers:
    @patch('app.scrapers.driver_scraper.create_session')
    @patch('app.scrapers.driver_scraper.get_all_drivers_from_data')
    def test_handles_no_drivers(self, mock_get_drivers, mock_create_session):
        # No drivers found
        mock_get_drivers.return_value = []
        mock_session = Mock()
        mock_create_session.return_value = mock_session

        # Call with None so function will call create_session()
        scrape_drivers(None)  # Should not raise

        mock_get_drivers.assert_called_once()
        # Because function returns before try/finally, session.close is NOT called here.
        # If you want to assert session was created:
        mock_create_session.assert_called_once()

    @patch('app.scrapers.driver_scraper.create_session')
    @patch('app.scrapers.driver_scraper.os.makedirs')
    @patch('app.scrapers.driver_scraper.scrape_driver_profile')
    @patch('app.scrapers.driver_scraper.get_all_drivers_from_data')
    def test_scrapes_all_drivers(self, mock_get_drivers, mock_scrape_profile,
                                 mock_makedirs, mock_create_session):
        mock_get_drivers.return_value = ['Lewis Hamilton', 'Max Verstappen']

        # scrape_driver_profile returns a dict per driver
        mock_scrape_profile.side_effect = [
            {'name': 'Lewis Hamilton', 'scraped': True},
            {'name': 'Max Verstappen', 'scraped': True}
        ]

        mock_session = Mock()
        mock_create_session.return_value = mock_session

        scrape_drivers(None)

        # We don't rely on order (set() used); just ensure both were called
        assert mock_scrape_profile.call_count == 2

        # session.close should be called once in the finally block
        mock_session.close.assert_called_once()

        # ensure profiles dir was ensured
        mock_makedirs.assert_called_once()

    @patch('app.scrapers.driver_scraper.create_session')
    @patch('app.scrapers.driver_scraper.os.makedirs')
    @patch('app.scrapers.driver_scraper.scrape_driver_profile')
    @patch('app.scrapers.driver_scraper.get_all_drivers_from_data')
    def test_closes_session_on_error(self, mock_get_drivers, mock_scrape_profile,
                                     mock_makedirs, mock_create_session):
        mock_get_drivers.return_value = ['Lewis Hamilton']

        # Make the per-driver scraper raise to simulate an error
        mock_scrape_profile.side_effect = Exception("scrapers error")
        mock_session = Mock()
        mock_create_session.return_value = mock_session

        with pytest.raises(Exception):
            scrape_drivers(None)

        # Even on error, session.close() must be called from finally
        mock_session.close.assert_called_once()
