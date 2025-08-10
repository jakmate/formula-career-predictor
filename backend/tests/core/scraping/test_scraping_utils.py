from bs4 import BeautifulSoup
from app.core.scraping.scraping_utils import remove_superscripts


def test_remove_superscripts_basic():
    html = '<td>Text with <sup>1</sup> citation</td>'
    soup = BeautifulSoup(html, 'lxml')
    cell = soup.find('td')

    result = remove_superscripts(cell)
    assert result == "Text with citation"


def test_remove_superscripts_multiple():
    html = '<td>Text <sup>1</sup> with <sup>2</sup> multiple <sup>3</sup> citations</td>'
    soup = BeautifulSoup(html, 'lxml')
    cell = soup.find('td')

    result = remove_superscripts(cell)
    assert result == "Text with multiple citations"


def test_remove_superscripts_nested_elements():
    html = '<td><span>Text</span> <sup>1</sup> <strong>bold</strong> <sup>2</sup></td>'
    soup = BeautifulSoup(html, 'lxml')
    cell = soup.find('td')

    result = remove_superscripts(cell)
    assert result == "Text bold"


def test_remove_superscripts_empty_cell():
    html = '<td></td>'
    soup = BeautifulSoup(html, 'lxml')
    cell = soup.find('td')

    result = remove_superscripts(cell)
    assert result == ""


def test_remove_superscripts_only_superscripts():
    html = '<td><sup>1</sup><sup>2</sup></td>'
    soup = BeautifulSoup(html, 'lxml')
    cell = soup.find('td')

    result = remove_superscripts(cell)
    assert result == ""


def test_remove_superscripts_no_superscripts():
    html = '<td>Plain text without citations</td>'
    soup = BeautifulSoup(html, 'lxml')
    cell = soup.find('td')

    result = remove_superscripts(cell)
    assert result == "Plain text without citations"


def test_remove_superscripts_complex_html():
    html = '''
    <td>
        <div>Section 1 <sup>a</sup></div>
        <p>Section 2 <sup>b</sup> with <em>emphasis</em></p>
        <span>Final <sup>c</sup> part</span>
    </td>
    '''
    soup = BeautifulSoup(html, 'lxml')
    cell = soup.find('td')

    result = remove_superscripts(cell)
    assert result == "Section 1 Section 2 with emphasis Final part"
