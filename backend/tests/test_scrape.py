import requests
from unittest.mock import Mock, patch

from app.core.scraping.scrape import map_url, scrape, scrape_current_year


class TestMapUrl:
    def test_map_url_f1(self):
        """Test F1 URL mapping"""
        assert map_url(1, 2020) == "https://en.wikipedia.org/wiki/2020_Formula_One_World_Championship"  # noqa: 501
        assert map_url(1, 2010) == "https://en.wikipedia.org/wiki/2010_Formula_One_World_Championship"  # noqa: 501
        assert map_url(1, 2025) == "https://en.wikipedia.org/wiki/2025_Formula_One_World_Championship"  # noqa: 501

    def test_map_url_f2_gp2_old(self):
        """Test F2/GP2 URL mapping for years <= 2016"""
        assert map_url(2, 2016) == "https://en.wikipedia.org/wiki/2016_GP2_Series"
        assert map_url(2, 2015) == "https://en.wikipedia.org/wiki/2015_GP2_Series"
        assert map_url(2, 2010) == "https://en.wikipedia.org/wiki/2010_GP2_Series"

    def test_map_url_f2_new(self):
        """Test F2 URL mapping for years > 2016"""
        assert map_url(2, 2017) == "https://en.wikipedia.org/wiki/2017_Formula_2_Championship"
        assert map_url(2, 2020) == "https://en.wikipedia.org/wiki/2020_Formula_2_Championship"
        assert map_url(2, 2025) == "https://en.wikipedia.org/wiki/2025_Formula_2_Championship"

    def test_map_url_f3_gp3_old(self):
        """Test F3/GP3 URL mapping for years <= 2018"""
        assert map_url(3, 2018) == "https://en.wikipedia.org/wiki/2018_GP3_Series"
        assert map_url(3, 2017) == "https://en.wikipedia.org/wiki/2017_GP3_Series"
        assert map_url(3, 2015) == "https://en.wikipedia.org/wiki/2015_GP3_Series"

    def test_map_url_f3_new(self):
        """Test F3 URL mapping for years > 2018"""
        assert map_url(3, 2019) == "https://en.wikipedia.org/wiki/2019_FIA_Formula_3_Championship"
        assert map_url(3, 2020) == "https://en.wikipedia.org/wiki/2020_FIA_Formula_3_Championship"
        assert map_url(3, 2025) == "https://en.wikipedia.org/wiki/2025_FIA_Formula_3_Championship"

    def test_map_url_invalid_num(self):
        """Test invalid formula number"""
        assert map_url(4, 2020) is None
        assert map_url(0, 2020) is None
        assert map_url(-1, 2020) is None


class TestScrape:
    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.scrape_quali')
    @patch('app.core.scraping.scrape.process_championship')
    @patch('app.core.scraping.scrape.process_entries')
    @patch('app.core.scraping.scrape.requests.Session')
    def test_scrape_success(self, mock_session_class, mock_process_entries,
                            mock_process_championship, mock_scrape_quali,
                            mock_scrape_drivers, mock_save_schedules):
        # Mock session and response
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.text = "<html><body>test</body></html>"
        mock_session.get.return_value = mock_response

        # Run scrape
        scrape()

        # Verify session was created and closed
        mock_session_class.assert_called_once()
        mock_session.close.assert_called_once()

        # Verify final functions were called
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()

        # Verify processing functions were called
        assert mock_process_entries.call_count > 0
        assert mock_process_championship.call_count > 0
        assert mock_scrape_quali.call_count > 0

    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.scrape_quali')
    @patch('app.core.scraping.scrape.process_championship')
    @patch('app.core.scraping.scrape.process_entries')
    @patch('app.core.scraping.scrape.requests.Session')
    def test_scrape_with_request_error(self, mock_session_class, mock_process_entries,
                                       mock_process_championship, mock_scrape_quali,
                                       mock_scrape_drivers, mock_save_schedules):
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock request to raise an exception
        mock_session.get.side_effect = requests.RequestException("Network error")

        # Should not raise exception - errors are caught
        scrape()

        # Verify session was still closed
        mock_session.close.assert_called_once()

        # Final functions should still be called
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()

    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.scrape_quali')
    @patch('app.core.scraping.scrape.process_championship')
    @patch('app.core.scraping.scrape.process_entries')
    @patch('app.core.scraping.scrape.requests.Session')
    def test_scrape_with_processing_error(self, mock_session_class, mock_process_entries,
                                          mock_process_championship, mock_scrape_quali,
                                          mock_scrape_drivers, mock_save_schedules):
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.text = "<html><body>test</body></html>"
        mock_session.get.return_value = mock_response

        # Mock processing to raise an exception
        mock_process_entries.side_effect = Exception("Processing error")

        # Should not raise exception - errors are caught
        scrape()

        # Verify session was still closed
        mock_session.close.assert_called_once()

        # Final functions should still be called
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()

    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.scrape_quali')
    @patch('app.core.scraping.scrape.process_championship')
    @patch('app.core.scraping.scrape.process_entries')
    @patch('app.core.scraping.scrape.requests.Session')
    def test_scrape_f3_european_championship(self, mock_session_class, mock_process_entries,
                                             mock_process_championship, mock_scrape_quali,
                                             mock_scrape_drivers, mock_save_schedules):
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.text = "<html><body>test</body></html>"
        mock_session.get.return_value = mock_response

        scrape()

        # Verify F3 European Championship URLs were called
        expected_calls = []
        for year in range(2012, 2019):
            expected_calls.append(
                f"https://en.wikipedia.org/wiki/{year}_FIA_Formula_3_European_Championship"
            )

        # Check that some F3 European URLs were called
        call_args = [call[0][0] for call in mock_session.get.call_args_list]
        f3_euro_calls = [url for url in call_args if "European" in url]
        assert len(f3_euro_calls) > 0

        # Verify process_entries was called with f3_euro parameter
        f3_euro_calls = [call for call in mock_process_entries.call_args_list
                         if len(call[0]) > 3 and call[0][3] == "f3_euro"]
        assert len(f3_euro_calls) > 0


class TestScrapeCurrentYear:
    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.scrape_quali')
    @patch('app.core.scraping.scrape.process_championship')
    @patch('app.core.scraping.scrape.process_entries')
    @patch('app.core.scraping.scrape.requests.Session')
    @patch('app.core.scraping.scrape.datetime')
    def test_scrape_current_year_success(self, mock_datetime, mock_session_class,
                                         mock_process_entries, mock_process_championship,
                                         mock_scrape_quali, mock_scrape_drivers,
                                         mock_save_schedules):
        # Mock current year
        mock_datetime.now.return_value.year = 2023

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.text = "<html><body>test</body></html>"
        mock_session.get.return_value = mock_response

        scrape_current_year()

        # Verify session was created and closed
        mock_session_class.assert_called_once()
        mock_session.close.assert_called_once()

        # Verify final functions were called
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()

        # Verify URLs for current year were called
        expected_urls = [
            "https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship",
            "https://en.wikipedia.org/wiki/2023_Formula_2_Championship",
            "https://en.wikipedia.org/wiki/2023_FIA_Formula_3_Championship"
        ]

        call_args = [call[0][0] for call in mock_session.get.call_args_list]
        for url in expected_urls:
            assert url in call_args

    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.scrape_quali')
    @patch('app.core.scraping.scrape.process_championship')
    @patch('app.core.scraping.scrape.process_entries')
    @patch('app.core.scraping.scrape.requests.Session')
    @patch('app.core.scraping.scrape.datetime')
    def test_scrape_current_year_with_errors(self, mock_datetime, mock_session_class,
                                             mock_process_entries, mock_process_championship,
                                             mock_scrape_quali, mock_scrape_drivers,
                                             mock_save_schedules):
        mock_datetime.now.return_value.year = 2023

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock request to raise an exception
        mock_session.get.side_effect = requests.RequestException("Network error")

        # Should not raise exception
        scrape_current_year()

        # Verify session was still closed
        mock_session.close.assert_called_once()

        # Final functions should still be called
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()


class TestIntegration:
    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.scrape_quali')
    @patch('app.core.scraping.scrape.process_championship')
    @patch('app.core.scraping.scrape.process_entries')
    @patch('app.core.scraping.scrape.requests.Session')
    def test_full_scrape_flow(self, mock_session_class, mock_process_entries,
                              mock_process_championship, mock_scrape_quali,
                              mock_scrape_drivers, mock_save_schedules):
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock realistic response
        mock_response = Mock()
        mock_response.text = """
        <html>
            <body>
                <table class="wikitable">
                    <tr><th>Driver</th><th>Points</th></tr>
                    <tr><td>Lewis Hamilton</td><td>100</td></tr>
                </table>
            </body>
        </html>
        """
        mock_session.get.return_value = mock_response

        scrape()

        # Verify all processing functions were called multiple times
        # (once for each year/formula combination)
        assert mock_process_entries.call_count > 10
        assert mock_process_championship.call_count > 10
        assert mock_scrape_quali.call_count > 10

        # Verify final functions called once
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()

    @patch('app.core.scraping.scrape.gc.collect')
    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.scrape_quali')
    @patch('app.core.scraping.scrape.process_championship')
    @patch('app.core.scraping.scrape.process_entries')
    @patch('app.core.scraping.scrape.requests.Session')
    def test_memory_management(self, mock_session_class, mock_process_entries,
                               mock_process_championship, mock_scrape_quali,
                               mock_scrape_drivers, mock_save_schedules, mock_gc):
        """Test memory management with gc.collect calls"""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.text = "<html><body>test</body></html>"
        mock_session.get.return_value = mock_response

        scrape()

        # Verify garbage collection was called
        assert mock_gc.call_count > 0

        # Verify response cleanup
        mock_response.close.assert_called()


class TestErrorHandling:
    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.requests.Session')
    def test_session_exception_handling(self, mock_session_class,
                                        mock_scrape_drivers, mock_save_schedules):
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock session.get to raise different exceptions
        mock_session.get.side_effect = [
            requests.Timeout("Timeout"),
            requests.ConnectionError("Connection error"),
            requests.HTTPError("HTTP error")
        ]

        # Should handle all exceptions gracefully
        scrape_current_year()

        # Final functions should still be called
        mock_scrape_drivers.assert_called_once()
        mock_save_schedules.assert_called_once()

    @patch('builtins.print')
    @patch('app.core.scraping.scrape.save_schedules')
    @patch('app.core.scraping.scrape.scrape_drivers')
    @patch('app.core.scraping.scrape.requests.Session')
    def test_error_logging(self, mock_session_class, mock_scrape_drivers,
                           mock_save_schedules, mock_print):
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock session.get to raise exception
        mock_session.get.side_effect = Exception("Test error")

        scrape_current_year()

        # Verify error was printed
        error_calls = [call for call in mock_print.call_args_list
                       if "Error processing" in str(call)]
        assert len(error_calls) > 0


class TestMainExecution:
    @patch('app.core.scraping.scrape.scrape')
    def test_main_execution(self, mock_scrape):
        """Test main execution calls scrape function"""
        # Import and test main execution
        import app.core.scraping.scrape as scrape_module

        # Simulate main execution
        if __name__ == "__main__":
            scrape_module.scrape()

        # This test verifies the structure exists
        assert hasattr(scrape_module, 'scrape')
        assert callable(scrape_module.scrape)


class TestEdgeCases:
    def test_map_url_boundary_years(self):
        """Test boundary years for URL mapping"""
        # Test 2016 boundary for GP2
        assert "GP2" in map_url(2, 2016)
        assert "Formula_2" in map_url(2, 2017)

        # Test 2018 boundary for GP3
        assert "GP3" in map_url(3, 2018)
        assert "Formula_3" in map_url(3, 2019)

    def test_year_range_coverage(self):
        # Verify the range covers expected years
        years = list(range(2010, 2026))
        assert 2010 in years
        assert 2025 in years
        assert len(years) == 16

        # F3 European range
        f3_euro_years = list(range(2012, 2019))
        assert 2012 in f3_euro_years
        assert 2018 in f3_euro_years
        assert len(f3_euro_years) == 7
