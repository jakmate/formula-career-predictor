from unittest.mock import Mock, patch

import pytest
from app.scrapers.scrape import map_url, scrape, scrape_current_year, scrape_wiki


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


class TestScrapeWiki:
    """Test scrape_wiki function."""

    @patch('app.scrapers.scrape.create_session')
    @patch('app.scrapers.scrape.safe_request')
    @patch('app.scrapers.scrape.process_entries')
    @patch('app.scrapers.scrape.process_championship')
    @patch('app.scrapers.scrape.scrape_quali')
    def test_scrape_wiki_successful(
        self, mock_quali, mock_championship, mock_entries,
        mock_request, mock_session
    ):
        """Test successful scraping for a year range."""
        mock_sess = Mock()
        mock_session.return_value = mock_sess

        mock_response = Mock()
        mock_response.text = "<html><body></body></html>"
        mock_request.return_value = mock_response

        scrape_wiki(mock_sess, formulas=[1], start_year=2020, end_year=2021)

        assert mock_request.call_count == 1
        assert mock_entries.call_count == 1
        assert mock_championship.call_count == 2  # Teams and Drivers
        assert mock_quali.call_count == 1

    @patch('app.scrapers.scrape.create_session')
    @patch('app.scrapers.scrape.safe_request')
    def test_scrape_wiki_request_failure(self, mock_request, mock_session):
        """Test handling of request failures."""
        mock_sess = Mock()
        mock_session.return_value = mock_sess
        mock_request.return_value = None  # Simulate failure

        scrape_wiki(mock_sess, formulas=[1], start_year=2020, end_year=2021)

        assert mock_request.call_count == 1

    @patch('app.scrapers.scrape.create_session')
    @patch('app.scrapers.scrape.safe_request')
    @patch('app.scrapers.scrape.process_entries')
    def test_scrape_wiki_processing_error(
        self, mock_entries, mock_request, mock_session
    ):
        """Test handling of processing errors."""
        mock_sess = Mock()
        mock_session.return_value = mock_sess

        mock_response = Mock()
        mock_response.text = "<html><body></body></html>"
        mock_request.return_value = mock_response

        mock_entries.side_effect = Exception("Processing error")

        # Should not raise, just log error
        scrape_wiki(mock_sess, formulas=[1], start_year=2020, end_year=2021)


class TestScrapeFunctions:
    """Test main scrape functions."""

    @patch('app.scrapers.scrape.create_session')
    @patch('app.scrapers.scrape.scrape_wiki')
    @patch('app.scrapers.scrape.scrape_drivers')
    @patch('app.scrapers.scrape.scrape_schedules')
    def test_scrape(self, mock_schedules, mock_drivers, mock_wiki, mock_session):
        """Test scrape function calls all scrapers."""
        mock_sess = Mock()
        mock_session.return_value = mock_sess

        scrape()

        mock_wiki.assert_called_once_with(mock_sess)
        mock_drivers.assert_called_once_with(mock_sess)
        mock_schedules.assert_called_once_with(mock_sess)
        mock_sess.close.assert_called_once()

    @patch('app.scrapers.scrape.create_session')
    @patch('app.scrapers.scrape.scrape_wiki')
    @patch('app.scrapers.scrape.scrape_drivers')
    @patch('app.scrapers.scrape.scrape_schedules')
    @patch('app.scrapers.scrape.CURRENT_YEAR', 2024)
    def test_scrape_current_year(
        self, mock_schedules, mock_drivers, mock_wiki, mock_session
    ):
        """Test scrape_current_year only scrapes current year."""
        mock_sess = Mock()
        mock_session.return_value = mock_sess

        scrape_current_year()

        mock_wiki.assert_called_once_with(mock_sess, start_year=2024)
        mock_drivers.assert_called_once_with(mock_sess)
        mock_schedules.assert_called_once_with(mock_sess)
        mock_sess.close.assert_called_once()

    @patch('app.scrapers.scrape.create_session')
    @patch('app.scrapers.scrape.scrape_wiki')
    def test_scrape_closes_session_on_error(self, mock_wiki, mock_session):
        """Test session closes even if scraping fails."""
        mock_sess = Mock()
        mock_session.return_value = mock_sess
        mock_wiki.side_effect = Exception("Scrape failed")

        with pytest.raises(Exception):
            scrape()

        mock_sess.close.assert_called_once()
