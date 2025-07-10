def remove_superscripts(cell):
    """Clean cell text by removing sup elements and extracting clean text"""
    # Remove all sup elements (citations, footnotes, etc.)
    for sup in cell.find_all("sup"):
        sup.decompose()

    # Get clean text
    return cell.get_text(strip=True)
