import re


def remove_citations(text):
    """Remove Wikipedia-style citations (e.g., [1], [a], [Note]) from text."""
    return re.sub(r'\[[^\]]+\]', '', text)
