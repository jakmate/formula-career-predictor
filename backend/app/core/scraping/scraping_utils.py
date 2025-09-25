import os
import requests
import time

from app.config import DATA_DIR


def create_session():
    """Create a session with a single, stable user agent"""
    session = requests.Session()

    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36', # noqa: 501
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })

    return session


def safe_request(session, url, max_retries=3, base_delay=1):
    """Make a request with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"403 error on attempt {attempt + 1} for {url}")
                if attempt < max_retries - 1:
                    wait_time = 2 + (attempt * 1)
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Final 403 error for {url} - skipping")
                    return None
            else:
                raise
        except Exception as e:
            print(f"Error on attempt {attempt + 1} for {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (attempt + 1))
            else:
                return None

    return None


def remove_superscripts(cell, preserve_spaces=True):
    """Clean cell text by removing sup elements and extracting clean text"""
    # Remove all sup elements (citations, footnotes, etc.)
    for sup in cell.find_all("sup"):
        sup.decompose()

    # Get clean text with or without spaces between elements
    separator = ' ' if preserve_spaces else ''
    return cell.get_text(separator=separator, strip=True)


def create_output_file(series, year, filename):
    """Create output directory and file path."""
    dir_path = os.path.join(DATA_DIR, f"F{series}", str(year))
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, filename)
