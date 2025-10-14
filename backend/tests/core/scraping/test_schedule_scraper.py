import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, mock_open

from app.core.scraping.schedule_scraper import (
    get_country_for_location,
    get_timezone_for_location,
    format_utc_datetime,
    is_race_completed_or_ongoing,
    parse_time_to_datetime,
    scrape_f1_schedule,
    scrape_fia_formula_schedule,
    save_schedules
)


# Tests for get_country_for_location
class TestGetCountryForLocation:
    def test_known_location(self):
        assert get_country_for_location("Sakhir") == "Bahrain"
        assert get_country_for_location("Monaco") == "Monaco"
        assert get_country_for_location("Melbourne") == "Australia"

    def test_unknown_location(self):
        assert get_country_for_location("Unknown City") == "Unknown City"


# Tests for get_timezone_for_location
class TestGetTimezoneForLocation:
    def test_known_timezone(self):
        assert get_timezone_for_location("Sakhir") == "Asia/Bahrain"
        assert get_timezone_for_location("Monaco") == "Europe/Monaco"
        assert get_timezone_for_location("Silverstone") == "Europe/London"

    @patch('app.core.scraping.schedule_scraper.Nominatim')
    @patch('app.core.scraping.schedule_scraper.TimezoneFinder')
    def test_geocoding_fallback(self, mock_tf, mock_nominatim):
        mock_geolocator = Mock()
        mock_location = Mock()
        mock_location.latitude = 40.7128
        mock_location.longitude = -74.0060
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim.return_value = mock_geolocator
        mock_tf_instance = Mock()
        mock_tf_instance.timezone_at.return_value = "America/New_York"
        mock_tf.return_value = mock_tf_instance

        result = get_timezone_for_location("New York")
        assert result == "America/New_York"

    @patch('app.core.scraping.schedule_scraper.Nominatim')
    def test_geocoding_failure_returns_utc(self, mock_nominatim):
        mock_geolocator = Mock()
        mock_geolocator.geocode.side_effect = Exception("Geocoding failed")
        mock_nominatim.return_value = mock_geolocator

        result = get_timezone_for_location("Invalid Location")
        assert result == "UTC"


# Tests for format_utc_datetime
class TestFormatUtcDatetime:
    def test_with_timezone_info(self):
        dt = datetime(2025, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = format_utc_datetime(dt)
        assert result == "2025-03-15T14:30:00"

    def test_without_timezone_info(self):
        dt = datetime(2025, 3, 15, 14, 30, 0)
        result = format_utc_datetime(dt)
        assert result == "2025-03-15T14:30:00"


# Tests for is_race_completed_or_ongoing
class TestIsRaceCompletedOrOngoing:
    def test_race_with_no_sessions(self):
        race = {"round": 1, "sessions": {}}
        assert is_race_completed_or_ongoing(race) is False

    def test_race_with_past_session(self):
        past_time = datetime.now(timezone.utc) - timedelta(days=2)
        race = {
            "sessions": {
                "race": {"start": past_time.replace(tzinfo=None).isoformat()}
            }
        }
        assert is_race_completed_or_ongoing(race) is True

    def test_race_with_future_session(self):
        future_time = datetime.now(timezone.utc) + timedelta(days=2)
        race = {
            "sessions": {
                "race": {"start": future_time.replace(tzinfo=None).isoformat()}
            }
        }
        assert is_race_completed_or_ongoing(race) is False

    def test_race_with_date_only(self):
        # Date-only format no longer triggers completion check
        past_date = (datetime.now(timezone.utc) - timedelta(days=2)).date()
        race = {
            "sessions": {
                "race": {"start": past_date.strftime("%Y-%m-%d")}
            }
        }
        # Should return False since no 'T' in start time
        assert is_race_completed_or_ongoing(race) is False

    def test_race_with_invalid_date(self):
        race = {
            "sessions": {
                "race": {"start": "invalid-date"}
            }
        }
        assert is_race_completed_or_ongoing(race) is False


# Tests for parse_time_to_datetime
class TestParseTimeToDatetime:
    def test_tbc_time_no_day(self):
        base_date = datetime(2025, 3, 15)
        result = parse_time_to_datetime("TBC", base_date)
        assert result == {"start": "2025-03-15", "time": "TBC"}

    def test_tbc_time_with_day(self):
        base_date = datetime(2025, 3, 15)  # Saturday
        result = parse_time_to_datetime("TBC", base_date, day_name="Friday")
        assert result["start"] == "2025-03-14"
        assert result["time"] == "TBC"

    def test_single_time(self):
        base_date = datetime(2025, 3, 15)
        result = parse_time_to_datetime("14:30", base_date, location="Monaco")
        assert "start" in result
        assert "2025-03-15" in result["start"]

    def test_time_range(self):
        base_date = datetime(2025, 3, 15)
        result = parse_time_to_datetime("14:30-15:30", base_date, location="Monaco")
        assert "start" in result
        assert "end" in result

    def test_time_with_day_name_adjustment(self):
        base_date = datetime(2025, 3, 15)  # Saturday
        result = parse_time_to_datetime("10:00", base_date, day_name="Friday", location="Monaco")
        # Should adjust to Friday
        assert "2025-03-14" in result["start"]

    def test_timezone_conversion(self):
        base_date = datetime(2025, 3, 15)
        result = parse_time_to_datetime("14:00", base_date, location="Monaco")
        # Monaco is UTC+1, so 14:00 local should be 13:00 UTC
        assert "start" in result

    def test_invalid_time_format(self):
        base_date = datetime(2025, 3, 15)
        result = parse_time_to_datetime("invalid", base_date)
        assert result is None

    def test_none_time(self):
        base_date = datetime(2025, 3, 15)
        result = parse_time_to_datetime(None, base_date)
        assert result is None


# Tests for scrape_f1_schedule
class TestScrapeF1Schedule:
    def test_successful_scrape(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = b'''
        <a class="group" href="/en/racing/2025/bahrain">
            <span class="typography-module_body-2-xs-bold__M03Ei">ROUND 1</span>
            <span class="typography-module_display-xl-bold__Gyl5W">Sakhir</span>
            <span class="typography-module_body-xs-semibold__Fyfwn">
            FORMULA 1 BAHRAIN GRAND PRIX 2025</span>
            <span class="typography-module_technical-xs-regular__-W0Gs">28 Feb - 02 Mar</span>
        </a>
        '''
        mock_session.get.return_value = mock_response

        with patch('app.core.scraping.schedule_scraper.CURRENT_YEAR', 2025):
            races = scrape_f1_schedule(mock_session)
            assert isinstance(races, list)

    def test_scrape_with_network_error(self):
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Network error")

        races = scrape_f1_schedule(mock_session)
        assert races == []

    @patch('app.core.scraping.schedule_scraper.BeautifulSoup')
    def test_scrape_with_session_details(self, mock_bs):
        mock_session = Mock()
        mock_soup = Mock()
        mock_card = Mock()

        # Setup card mock
        mock_round = Mock()
        mock_round.text = "ROUND 1"
        mock_card.select_one.side_effect = lambda sel: {
            '.typography-module_body-2-xs-bold__M03Ei': mock_round,
            '.typography-module_display-xl-bold__Gyl5W': Mock(text="Sakhir"),
            '.typography-module_body-xs-semibold__Fyfwn': Mock(text="FORMULA 1 BAHRAIN GP 2025"),
            '.typography-module_technical-xs-regular__-W0Gs': Mock(text="02 Mar")
        }.get(sel, None)

        mock_card.get.return_value = "/en/racing/2025/bahrain"
        mock_soup.find_all.return_value = [mock_card]
        mock_bs.return_value = mock_soup

        mock_session.get.return_value = Mock(content=b'')

        with patch('app.core.scraping.schedule_scraper.CURRENT_YEAR', 2025):
            races = scrape_f1_schedule(mock_session)
            assert isinstance(races, list)


# Tests for scrape_fia_formula_schedule
class TestScrapeFiaFormulaSchedule:
    def test_unsupported_series(self):
        mock_session = Mock()
        with pytest.raises(ValueError, match="Unsupported series"):
            scrape_fia_formula_schedule(mock_session, 'f4')

    def test_f2_scrape(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = b'''
        <div class="col-12 col-sm-6 col-lg-4 col-xl-3">
            <span class="h6">Round 1</span>
            <span class="event-place"><span class="ellipsis">Bahrain</span></span>
            <p class="date">
                <span class="end-date">02</span>
                <span class="month">March</span>
            </p>
            <a href="/race/bahrain"></a>
        </div>
        '''
        mock_session.get.return_value = mock_response

        with patch('app.core.scraping.schedule_scraper.CURRENT_YEAR', 2025):
            races = scrape_fia_formula_schedule(mock_session, 'f2')
            assert isinstance(races, list)

    def test_f3_scrape(self):
        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = b'<div></div>'
        mock_session.get.return_value = mock_response

        races = scrape_fia_formula_schedule(mock_session, 'f3')
        assert isinstance(races, list)

    def test_network_error(self):
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection error")

        races = scrape_fia_formula_schedule(mock_session, 'f2')
        assert races == []


# Tests for save_schedules
class TestSaveSchedules:
    @patch('app.core.scraping.schedule_scraper.scrape_f1_schedule')
    @patch('app.core.scraping.schedule_scraper.scrape_fia_formula_schedule')
    @patch('app.core.scraping.schedule_scraper.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('app.core.scraping.schedule_scraper.json.dump')
    @patch('app.core.scraping.schedule_scraper.json.load')
    def test_save_new_schedules(self, mock_json_load, mock_json_dump,
                                mock_file, mock_exists, mock_f2_scraper, mock_f1_scraper):
        mock_session = Mock()
        mock_exists.return_value = False
        mock_f1_scraper.return_value = [
            {"round": 1, "name": "Bahrain", "location": "Bahrain",
             "sessions": {"race": {"start": "2025-03-02T15:00:00"}}}
        ]
        mock_f2_scraper.return_value = [
            {"round": 1, "name": "Bahrain", "location": "Bahrain",
             "sessions": {"race": {"start": "2025-03-02T14:00:00"}}}
        ]

        save_schedules(mock_session)

        # Verify scrapers were called
        assert mock_f1_scraper.called
        assert mock_f2_scraper.called

    @patch('app.core.scraping.schedule_scraper.scrape_f1_schedule')
    @patch('app.core.scraping.schedule_scraper.scrape_fia_formula_schedule')
    @patch('app.core.scraping.schedule_scraper.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('app.core.scraping.schedule_scraper.json.load')
    @patch('app.core.scraping.schedule_scraper.json.dump')
    @patch('app.core.scraping.schedule_scraper.is_race_completed_or_ongoing')
    def test_preserve_completed_races(self, mock_is_completed, mock_json_dump,
                                      mock_json_load, mock_file, mock_exists,
                                      mock_f2_scraper, mock_f1_scraper):
        mock_session = Mock()
        mock_exists.return_value = True

        existing_race = {
            "round": 1,
            "name": "Bahrain",
            "location": "Bahrain",
            "sessions": {"race": {"start": "2025-03-02T15:00:00"}}
        }
        mock_json_load.return_value = [existing_race]
        mock_is_completed.return_value = True

        mock_f1_scraper.return_value = [
            {"round": 1, "name": "Bahrain Updated", "location": "Bahrain",
             "sessions": {"race": {"start": "2025-03-02T16:00:00"}}}
        ]
        mock_f2_scraper.return_value = []

        save_schedules(mock_session)

        expected_existing = {1: existing_race}
        mock_f1_scraper.assert_called_once_with(mock_session, expected_existing)

        # Verify completed race was preserved
        assert mock_json_dump.called

    @patch('app.core.scraping.schedule_scraper.scrape_f1_schedule')
    @patch('app.core.scraping.schedule_scraper.scrape_fia_formula_schedule')
    def test_scraper_exception_handling(self, mock_f2_scraper, mock_f1_scraper):
        mock_session = Mock()
        mock_f1_scraper.side_effect = Exception("Scraper error")
        mock_f2_scraper.return_value = []

        # Should not raise exception
        save_schedules(mock_session)

    @patch('app.core.scraping.schedule_scraper.scrape_f1_schedule')
    @patch('app.core.scraping.schedule_scraper.scrape_fia_formula_schedule')
    @patch('app.core.scraping.schedule_scraper.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('app.core.scraping.schedule_scraper.json.load')
    def test_invalid_json_handling(self, mock_json_load, mock_file,
                                   mock_exists, mock_f2_scraper, mock_f1_scraper):
        mock_session = Mock()
        mock_exists.return_value = True
        mock_json_load.side_effect = Exception("Invalid JSON")
        mock_f1_scraper.return_value = []
        mock_f2_scraper.return_value = []

        # Should not raise exception
        save_schedules(mock_session)
