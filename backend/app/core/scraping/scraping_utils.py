def remove_superscripts(cell, preserve_spaces=True):
    """Clean cell text by removing sup elements and extracting clean text"""
    # Remove all sup elements (citations, footnotes, etc.)
    for sup in cell.find_all("sup"):
        sup.decompose()

    # Get clean text with or without spaces between elements
    separator = ' ' if preserve_spaces else ''
    return cell.get_text(separator=separator, strip=True)
