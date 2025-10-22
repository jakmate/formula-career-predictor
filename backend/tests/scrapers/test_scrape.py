from unittest.mock import Mock, patch
from app.scrapers.scrape import map_url, scrape, scrape_current_year


class TestMapUrl:
    """Test the map_url function with various inputs."""

    def test_map_url_f1(self):
        """Test F1 URL mapping."""
        assert map_url(1, 2020) == "https://en.wikipedia.org/wiki/2020_Formula_One_World_Championship" # noqa: 501
        assert map_url(1, 2010) == "https://en.wikipedia.org/wiki/2010_Formula_One_World_Championship" # noqa: 501

    def test_map_url_f2_after_2016(self):
        """Test F2 URL mapping for years after 2016."""
        assert map_url(2, 2020) == "https://en.wikipedia.org/wiki/2020_Formula_2_Championship"
        assert map_url(2, 2017) == "https://en.wikipedia.org/wiki/2017_Formula_2_Championship"

    def test_map_url_gp2_2016_and_before(self):
        """Test GP2 URL mapping for 2016 and earlier."""
        assert map_url(2, 2016) == "https://en.wikipedia.org/wiki/2016_GP2_Series"
        assert map_url(2, 2010) == "https://en.wikipedia.org/wiki/2010_GP2_Series"

    def test_map_url_f3_after_2018(self):
        """Test F3 URL mapping for years after 2018."""
        assert map_url(3, 2020) == "https://en.wikipedia.org/wiki/2020_FIA_Formula_3_Championship"
        assert map_url(3, 2019) == "https://en.wikipedia.org/wiki/2019_FIA_Formula_3_Championship"

    def test_map_url_gp3_2018_and_before(self):
        """Test GP3 URL mapping for 2018 and earlier."""
        assert map_url(3, 2018) == "https://en.wikipedia.org/wiki/2018_GP3_Series"
        assert map_url(3, 2010) == "https://en.wikipedia.org/wiki/2010_GP3_Series"

    def test_map_url_invalid_num(self):
        """Test invalid formula number returns None."""
        assert map_url(4, 2020) is None
        assert map_url(0, 2020) is None
        assert map_url(-1, 2020) is None


class TestScrape:
    """Test the main scrape function."""

    @patch('app.scrapers.scrape.save_schedules')
    @patch('app.scrapers.scrape.scrape_drivers')
    @patch('app.scrapers.scrape.scrape_quali')
    @patch('app.scrapers.scrape.process_championship')
    @patch('app.scrapers.scrape.process_entries')
    @patch('app.scrapers.scrape.safe_request')
    @patch('app.scrapers.scrape.create_session')
    def test_scrape_success(self, mock_create_session, mock_safe_request,
                            mock_process_entries, mock_process_championship,
                            mock_scrape_quali, mock_scrape_drivers, mock_save_schedules):
        """Test successful scrapers of all series."""
        # Setup mocks
        mock_session = Mock()
        mock_create_session.return_value = mock_session

        mock_response = Mock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_safe_request.return_value = mock_response

        # Run scrape function with limited year range for testing
        with patch('app.scrapers.scrape.range', return_value=[2020, 2021]):
            scrape()

        # Verify session creation and closure
        mock_create_session.assert_called_once()
        mock_session.close.assert_called_once()

        # Verify final functions called
        mock_scrape_drivers.assert_called()
        mock_save_schedules.assert_called()

    @patch('app.scrapers.scrape.save_schedules')
    @patch('app.scrapers.scrape.scrape_drivers')
    @patch('app.scrapers.scrape.safe_request')
    @patch('app.scrapers.scrape.create_session')
    def test_scrape_request_failure(self, mock_create_session, mock_safe_request,
                                    mock_scrape_drivers, mock_save_schedules):
        """Test scrape behavior when safe_request returns None."""
        mock_session = Mock()
        mock_create_session.return_value = mock_session
        mock_safe_request.return_value = None

        with patch('app.scrapers.scrape.range', return_value=[2020]):
            scrape()

        mock_session.close.assert_called_once()
        mock_scrape_drivers.assert_called()
        mock_save_schedules.assert_called()

    @patch('app.scrapers.scrape.save_schedules')
    @patch('app.scrapers.scrape.scrape_drivers')
    @patch('app.scrapers.scrape.process_entries')
    @patch('app.scrapers.scrape.safe_request')
    @patch('app.scrapers.scrape.create_session')
    def test_scrape_processing_exception(self, mock_create_session, mock_safe_request,
                                         mock_process_entries, mock_scrape_drivers,
                                         mock_save_schedules):
        """Test scrape behavior when processing raises an exception."""
        mock_session = Mock()
        mock_create_session.return_value = mock_session

        mock_response = Mock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_safe_request.return_value = mock_response

        # Make process_entries raise an exception
        mock_process_entries.side_effect = Exception("Processing error")

        with patch('app.scrapers.scrape.range', return_value=[2020]):
            scrape()

        mock_session.close.assert_called_once()
        mock_scrape_drivers.assert_called()
        mock_save_schedules.assert_called()


class TestScrapeCurrentYear:
    """Test the scrape_current_year function."""

    @patch('app.scrapers.scrape.datetime')
    @patch('app.scrapers.scrape.save_schedules')
    @patch('app.scrapers.scrape.scrape_drivers')
    @patch('app.scrapers.scrape.scrape_quali')
    @patch('app.scrapers.scrape.process_championship')
    @patch('app.scrapers.scrape.process_entries')
    @patch('app.scrapers.scrape.safe_request')
    @patch('app.scrapers.scrape.create_session')
    def test_scrape_current_year_success(self, mock_create_session, mock_safe_request,
                                         mock_process_entries, mock_process_championship,
                                         mock_scrape_quali, mock_scrape_drivers,
                                         mock_save_schedules, mock_datetime):
        """Test successful scrapers of current year."""
        # Mock current year
        mock_datetime.now.return_value.year = 2024

        # Setup mocks
        mock_session = Mock()
        mock_create_session.return_value = mock_session

        mock_response = Mock()
        mock_response.text = "<html><body>Current year content</body></html>"
        mock_safe_request.return_value = mock_response

        scrape_current_year()

        # Verify session handling
        mock_create_session.assert_called_once()
        mock_session.close.assert_called_once()

        # Verify all three series processed
        assert mock_safe_request.call_count == 3
        assert mock_process_entries.call_count == 3

        # Verify final functions called
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()

    @patch('app.scrapers.scrape.datetime')
    @patch('app.scrapers.scrape.save_schedules')
    @patch('app.scrapers.scrape.scrape_drivers')
    @patch('app.scrapers.scrape.safe_request')
    @patch('app.scrapers.scrape.create_session')
    def test_scrape_current_year_request_failure(self, mock_create_session,
                                                 mock_safe_request, mock_scrape_drivers,
                                                 mock_save_schedules, mock_datetime):
        """Test current year scrapers with request failures."""
        mock_datetime.now.return_value.year = 2024

        mock_session = Mock()
        mock_create_session.return_value = mock_session
        mock_safe_request.return_value = None

        scrape_current_year()

        mock_session.close.assert_called_once()
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()

    @patch('app.scrapers.scrape.datetime')
    @patch('app.scrapers.scrape.save_schedules')
    @patch('app.scrapers.scrape.scrape_drivers')
    @patch('app.scrapers.scrape.process_entries')
    @patch('app.scrapers.scrape.safe_request')
    @patch('app.scrapers.scrape.create_session')
    def test_scrape_current_year_processing_exception(self, mock_create_session,
                                                      mock_safe_request, mock_process_entries,
                                                      mock_scrape_drivers, mock_save_schedules,
                                                      mock_datetime):
        """Test current year scrapers with processing exceptions."""
        mock_datetime.now.return_value.year = 2024

        mock_session = Mock()
        mock_create_session.return_value = mock_session

        mock_response = Mock()
        mock_response.text = "<html><body>Content</body></html>"
        mock_safe_request.return_value = mock_response

        mock_process_entries.side_effect = Exception("Processing error")

        scrape_current_year()

        mock_session.close.assert_called_once()
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()
