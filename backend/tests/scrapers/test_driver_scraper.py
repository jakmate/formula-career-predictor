import json
from unittest.mock import Mock, patch, mock_open
import pandas as pd

from app.scrapers.driver_scraper import (
    get_driver_filename,
    needs_rescrape,
    search_wikidata_drivers,
    extract_nationality_from_result,
    extract_dob_from_result,
    save_profile,
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


class TestSearchWikidataDrivers:
    @patch('app.scrapers.driver_scraper.SPARQL_ENDPOINT', 'http://test.endpoint')
    def test_successful_batch_query(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': {
                'bindings': [
                    {
                        'nameMatch': {'value': 'Lewis Hamilton'},
                        'person': {'value': 'http://wikidata.org/entity/Q1'},
                        'dob': {'value': '1985-01-07T00:00:00Z'},
                        'nationalityLabel': {'value': 'United Kingdom'}
                    }
                ]
            }
        }
        mock_session.get.return_value = mock_response

        results = search_wikidata_drivers(['Lewis Hamilton'], mock_session)

        assert 'Lewis Hamilton' in results
        assert results['Lewis Hamilton']['person']['value'] == 'http://wikidata.org/entity/Q1'

    @patch('app.scrapers.driver_scraper.SPARQL_ENDPOINT', 'http://test.endpoint')
    def test_failed_query(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_session.get.return_value = mock_response

        results = search_wikidata_drivers(['Lewis Hamilton'], mock_session)

        assert results == {}

    @patch('app.scrapers.driver_scraper.SPARQL_ENDPOINT', 'http://test.endpoint')
    def test_query_exception(self):
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection error")

        results = search_wikidata_drivers(['Lewis Hamilton'], mock_session)

        assert results == {}

    @patch('app.scrapers.driver_scraper.SPARQL_ENDPOINT', 'http://test.endpoint')
    def test_batch_processing(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'results': {'bindings': []}}
        mock_session.get.return_value = mock_response

        # Test with 150 drivers to trigger multiple batches
        drivers = [f'Driver{i}' for i in range(150)]
        search_wikidata_drivers(drivers, mock_session, batch_size=100)

        # Should be called twice (batch 0-99, 100-149)
        assert mock_session.get.call_count == 2


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
    @patch('app.scrapers.driver_scraper.os.makedirs')
    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.search_wikidata_drivers')
    @patch('app.scrapers.driver_scraper.save_profile')
    def test_no_drivers_found(self, mock_save, mock_search, mock_exists,
                              mock_makedirs, mock_get_drivers, mock_session):
        mock_get_drivers.return_value = []
        mock_session.return_value = Mock()

        scrape_drivers()

        mock_search.assert_not_called()
        mock_save.assert_not_called()

    @patch('app.scrapers.driver_scraper.create_session')
    @patch('app.scrapers.driver_scraper.get_all_drivers_from_data')
    @patch('app.scrapers.driver_scraper.os.makedirs')
    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.search_wikidata_drivers')
    @patch('app.scrapers.driver_scraper.save_profile')
    @patch('app.scrapers.driver_scraper.PROFILES_DIR', '/profiles')
    def test_new_driver_no_results(self, mock_save, mock_search, mock_exists,
                                   mock_makedirs, mock_get_drivers, mock_session):
        mock_get_drivers.return_value = ['New Driver']
        mock_exists.return_value = False
        mock_search.return_value = {}
        session = Mock()
        mock_session.return_value = session

        scrape_drivers()

        # Should save profile with scraped=False
        assert mock_save.called
        saved_profile = mock_save.call_args[0][1]
        assert saved_profile['scraped'] is False
        assert saved_profile['name'] == 'New Driver'

    @patch('app.scrapers.driver_scraper.create_session')
    @patch('app.scrapers.driver_scraper.get_all_drivers_from_data')
    @patch('app.scrapers.driver_scraper.os.makedirs')
    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.search_wikidata_drivers')
    @patch('app.scrapers.driver_scraper.save_profile')
    @patch('app.scrapers.driver_scraper.PROFILES_DIR', '/profiles')
    def test_new_driver_with_results(self, mock_save, mock_search, mock_exists,
                                     mock_makedirs, mock_get_drivers, mock_session):
        mock_get_drivers.return_value = ['Lewis Hamilton']
        mock_exists.return_value = False
        mock_search.return_value = {
            'Lewis Hamilton': {
                'person': {'value': 'http://wikidata.org/entity/Q1'},
                'dob': {'value': '1985-01-07T00:00:00Z'},
                'nationalityLabel': {'value': 'United Kingdom'}
            }
        }
        session = Mock()
        mock_session.return_value = session

        scrape_drivers()

        saved_profile = mock_save.call_args[0][1]
        assert saved_profile['scraped'] is True
        assert saved_profile['dob'] == '1985-01-07'
        assert saved_profile['wikidata_id'] == 'Q1'

    @patch('app.scrapers.driver_scraper.create_session')
    @patch('app.scrapers.driver_scraper.get_all_drivers_from_data')
    @patch('app.scrapers.driver_scraper.os.makedirs')
    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.search_wikidata_drivers')
    @patch('app.scrapers.driver_scraper.save_profile')
    @patch('builtins.open', new_callable=mock_open)
    @patch('app.scrapers.driver_scraper.PROFILES_DIR', '/profiles')
    def test_existing_driver_needs_update(self, mock_file, mock_save, mock_search,
                                          mock_exists, mock_makedirs, mock_get_drivers,
                                          mock_session):
        mock_get_drivers.return_value = ['Lewis Hamilton']
        mock_exists.return_value = True

        existing_profile = json.dumps({
            'scraped': True,
            'dob': '1985-01-07',
            'nationality': 'UK'
        })
        mock_file.return_value.read.return_value = existing_profile

        mock_search.return_value = {
            'Lewis Hamilton': {
                'person': {'value': 'http://wikidata.org/entity/Q1'},
                'dob': {'value': '1985-01-07T00:00:00Z'},
                'nationalityLabel': {'value': 'United Kingdom'}
            }
        }
        session = Mock()
        mock_session.return_value = session

        scrape_drivers()

        # Should update profile due to nationality change
        assert mock_save.called

    @patch('app.scrapers.driver_scraper.get_all_drivers_from_data')
    def test_with_provided_session(self, mock_get_drivers):
        mock_get_drivers.return_value = []
        session = Mock()

        scrape_drivers(session=session)

        session.close.assert_not_called()

    @patch('app.scrapers.driver_scraper.create_session')
    @patch('app.scrapers.driver_scraper.get_all_drivers_from_data')
    @patch('app.scrapers.driver_scraper.os.makedirs')
    @patch('app.scrapers.driver_scraper.os.path.exists')
    @patch('app.scrapers.driver_scraper.search_wikidata_drivers')
    @patch('app.scrapers.driver_scraper.DRIVER_ALIASES', {'Test Driver': 'Aliased Driver'})
    def test_driver_alias_mapping(self, mock_search, mock_exists, mock_makedirs,
                                  mock_get_drivers, mock_session):
        mock_get_drivers.return_value = ['Test Driver']
        mock_exists.return_value = False
        mock_search.return_value = {}
        session = Mock()
        mock_session.return_value = session

        scrape_drivers()

        # Should search with aliased name
        call_args = mock_search.call_args[0][0]
        assert 'Aliased Driver' in call_args


class TestNeedsRescrape:
    def test_failed_scrape_needs_rescrape(self):
        existing = {'scraped': False, 'dob': None, 'nationality': None}
        new_data = {'dob': {'value': '1985-01-07T00:00:00Z'}}

        assert needs_rescrape(existing, new_data) is True

    def test_dob_changed(self):
        existing = {'scraped': True, 'dob': '1985-01-07', 'nationality': 'UK'}
        new_data = {'dob': {'value': '1986-01-07T00:00:00Z'}}

        assert needs_rescrape(existing, new_data) is True

    def test_nationality_changed(self):
        existing = {'scraped': True, 'dob': '1985-01-07', 'nationality': 'UK'}
        new_data = {
            'dob': {'value': '1985-01-07T00:00:00Z'},
            'nationalityLabel': {'value': 'France'}
        }

        assert needs_rescrape(existing, new_data) is True

    def test_no_changes(self):
        existing = {'scraped': True, 'dob': '1985-01-07', 'nationality': 'United Kingdom'}
        new_data = {
            'dob': {'value': '1985-01-07T00:00:00Z'},
            'nationalityLabel': {'value': 'United Kingdom'}
        }

        assert needs_rescrape(existing, new_data) is False
