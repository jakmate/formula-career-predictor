import random
import requests
import time

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]


def create_session():
    """Create a session with headers that mimic a real browser"""
    session = requests.Session()
    
    # Set headers to mimic a real browser
    session.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    return session

def remove_superscripts(cell, preserve_spaces=True):
    """Clean cell text by removing sup elements and extracting clean text"""
    # Remove all sup elements (citations, footnotes, etc.)
    for sup in cell.find_all("sup"):
        sup.decompose()

    # Get clean text with or without spaces between elements
    separator = ' ' if preserve_spaces else ''
    return cell.get_text(separator=separator, strip=True)

def safe_request(session, url, max_retries=3, base_delay=1):
    """Make a request with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            # Random delay to avoid being flagged
            delay = base_delay + random.uniform(0.5, 2.0)
            time.sleep(delay)
            
            # Rotate user agent on retries
            if attempt > 0:
                session.headers['User-Agent'] = random.choice(USER_AGENTS)
            
            response = session.get(url, timeout=15)
            response.raise_for_status()
            return response
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"403 error on attempt {attempt + 1} for {url}")
                if attempt < max_retries - 1:
                    # Exponential backoff for 403 errors
                    wait_time = (2 ** attempt) * 5 + random.uniform(1, 5)
                    print(f"Waiting {wait_time:.1f} seconds before retry...")
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
