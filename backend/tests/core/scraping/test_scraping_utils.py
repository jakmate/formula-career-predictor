import pytest
import requests
from unittest.mock import Mock, patch
from bs4 import BeautifulSoup
from app.core.scraping.scraping_utils import create_session, remove_superscripts, safe_request


class TestCreateSession:
    def test_create_session_returns_session(self):
        """Test that create_session returns a requests.Session object"""
        session = create_session()
        assert isinstance(session, requests.Session)

    def test_create_session_sets_headers(self):
        """Test that create_session sets required headers"""
        session = create_session()
        headers = session.headers

        assert 'User-Agent' in headers
        assert 'Mozilla/5.0' in headers['User-Agent']
        assert headers['Accept'] == 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' # noqa: 501
        assert headers['Accept-Language'] == 'en-US,en;q=0.5'
        assert headers['Accept-Encoding'] == 'gzip, deflate'
        assert headers['Connection'] == 'keep-alive'
        assert headers['Upgrade-Insecure-Requests'] == '1'


class TestRemoveSuperscripts:
    def test_remove_superscripts_basic(self):
        """Test removing sup elements from HTML"""
        html = '<div>Text<sup>1</sup> more text<sup>2</sup></div>'
        soup = BeautifulSoup(html, 'html.parser')
        cell = soup.find('div')

        result = remove_superscripts(cell)
        assert result == 'Text more text'

    def test_remove_superscripts_preserve_spaces_true(self):
        """Test preserve_spaces=True maintains spacing"""
        html = '<div><span>Text1</span><sup>1</sup><span>Text2</span></div>'
        soup = BeautifulSoup(html, 'html.parser')
        cell = soup.find('div')

        result = remove_superscripts(cell, preserve_spaces=True)
        assert result == 'Text1 Text2'

    def test_remove_superscripts_preserve_spaces_false(self):
        """Test preserve_spaces=False removes spacing"""
        html = '<div><span>Text1</span><sup>1</sup><span>Text2</span></div>'
        soup = BeautifulSoup(html, 'html.parser')
        cell = soup.find('div')

        result = remove_superscripts(cell, preserve_spaces=False)
        assert result == 'Text1Text2'

    def test_remove_superscripts_no_sup_elements(self):
        """Test with HTML containing no sup elements"""
        html = '<div>Just plain text</div>'
        soup = BeautifulSoup(html, 'html.parser')
        cell = soup.find('div')

        result = remove_superscripts(cell)
        assert result == 'Just plain text'

    def test_remove_superscripts_nested_sup(self):
        """Test with nested sup elements"""
        html = '<div>Text<sup>outer<sup>inner</sup></sup> end</div>'
        soup = BeautifulSoup(html, 'html.parser')
        cell = soup.find('div')

        result = remove_superscripts(cell)
        assert result == 'Text end'


class TestSafeRequest:
    def test_safe_request_successful(self):
        """Test successful request"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response

        result = safe_request(mock_session, 'http://test.com')

        assert result == mock_response
        mock_session.get.assert_called_once_with('http://test.com', timeout=15)
        mock_response.raise_for_status.assert_called_once()

    def test_safe_request_http_error_non_403(self):
        """Test HTTP error that's not 403"""
        mock_session = Mock()
        mock_response = Mock()
        http_error = requests.exceptions.HTTPError()
        http_error.response = Mock()
        http_error.response.status_code = 404
        mock_session.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = http_error

        with pytest.raises(requests.exceptions.HTTPError):
            safe_request(mock_session, 'http://test.com')

    @patch('time.sleep')
    def test_safe_request_403_error_retry_success(self, mock_sleep):
        """Test 403 error followed by successful retry"""
        mock_session = Mock()
        mock_response_fail = Mock()
        mock_response_success = Mock()

        http_error = requests.exceptions.HTTPError()
        http_error.response = Mock()
        http_error.response.status_code = 403

        mock_response_fail.raise_for_status.side_effect = http_error
        mock_response_success.raise_for_status.return_value = None

        mock_session.get.side_effect = [mock_response_fail, mock_response_success]

        result = safe_request(mock_session, 'http://test.com')

        assert result == mock_response_success
        assert mock_session.get.call_count == 2
        mock_sleep.assert_called_once_with(2)

    @patch('time.sleep')
    def test_safe_request_403_error_max_retries(self, mock_sleep):
        """Test 403 error exceeding max retries"""
        mock_session = Mock()
        mock_response = Mock()

        http_error = requests.exceptions.HTTPError()
        http_error.response = Mock()
        http_error.response.status_code = 403
        mock_response.raise_for_status.side_effect = http_error
        mock_session.get.return_value = mock_response

        result = safe_request(mock_session, 'http://test.com', max_retries=2)

        assert result is None
        assert mock_session.get.call_count == 2
        assert mock_sleep.call_count == 1  # Only sleeps between retries, not after final failure

    @patch('time.sleep')
    def test_safe_request_generic_exception_retry(self, mock_sleep):
        """Test generic exception with retry"""
        mock_session = Mock()
        mock_response = Mock()

        mock_session.get.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response
        ]
        mock_response.raise_for_status.return_value = None

        result = safe_request(mock_session, 'http://test.com')

        assert result == mock_response
        assert mock_session.get.call_count == 2
        mock_sleep.assert_called_once_with(1)  # base_delay * (attempt + 1) = 1 * 1

    @patch('time.sleep')
    def test_safe_request_generic_exception_max_retries(self, mock_sleep):
        """Test generic exception exceeding max retries"""
        mock_session = Mock()
        mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        result = safe_request(mock_session, 'http://test.com', max_retries=2)

        assert result is None
        assert mock_session.get.call_count == 2
        assert mock_sleep.call_count == 1

    def test_safe_request_custom_parameters(self):
        """Test safe_request with custom max_retries and base_delay"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response

        result = safe_request(mock_session, 'http://test.com', max_retries=5, base_delay=2)

        assert result == mock_response
        mock_session.get.assert_called_once_with('http://test.com', timeout=15)

    @patch('time.sleep')
    def test_safe_request_403_wait_times(self, mock_sleep):
        """Test 403 error wait times increase correctly"""
        mock_session = Mock()
        mock_response = Mock()

        http_error = requests.exceptions.HTTPError()
        http_error.response = Mock()
        http_error.response.status_code = 403
        mock_response.raise_for_status.side_effect = http_error
        mock_session.get.return_value = mock_response

        safe_request(mock_session, 'http://test.com', max_retries=3)

        # Should sleep twice: 2 seconds, then 3 seconds
        expected_calls = [pytest.approx(2), pytest.approx(3)]
        actual_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_calls == expected_calls
