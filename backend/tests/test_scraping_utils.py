from scraping.scraping_utils import remove_citations


class TestRemoveCitations:
    """Test cases for the remove_citations function."""

    def test_remove_single_citation(self):
        """Test removing a single citation from text."""
        text = "This is a test[1] sentence."
        expected = "This is a test sentence."
        assert remove_citations(text) == expected

    def test_remove_multiple_citations(self):
        """Test removing multiple citations from text."""
        text = "This[1] is a test[2] sentence[3]."
        expected = "This is a test sentence."
        assert remove_citations(text) == expected

    def test_remove_letter_citations(self):
        """Test removing letter-based citations."""
        text = "This is a test[a] sentence[b]."
        expected = "This is a test sentence."
        assert remove_citations(text) == expected

    def test_remove_note_citations(self):
        """Test removing note-style citations."""
        text = "This is a test[Note] sentence[Note 1]."
        expected = "This is a test sentence."
        assert remove_citations(text) == expected

    def test_remove_mixed_citations(self):
        """Test removing mixed citation types."""
        text = "Formula 1[1] is a racing[a] series[Note 1]."
        expected = "Formula 1 is a racing series."
        assert remove_citations(text) == expected

    def test_empty_string(self):
        """Test with empty string."""
        text = ""
        expected = ""
        assert remove_citations(text) == expected

    def test_no_citations(self):
        """Test text without citations."""
        text = "This is a test sentence."
        expected = "This is a test sentence."
        assert remove_citations(text) == expected

    def test_citation_with_spaces(self):
        """Test citations containing spaces."""
        text = "This is a test[citation with spaces] sentence."
        expected = "This is a test sentence."
        assert remove_citations(text) == expected

    def test_citation_with_special_characters(self):
        """Test citations with special characters."""
        text = "This is a test[1, 2] sentence[a-b]."
        expected = "This is a test sentence."
        assert remove_citations(text) == expected

    def test_nested_brackets_not_supported(self):
        """Test that nested brackets are not properly handled (expected limitation)."""
        text = "This is a test[citation [nested]] sentence."
        # The regex will match [citation [nested] - up to the first closing bracket
        expected = "This is a test] sentence."
        assert remove_citations(text) == expected

    def test_citation_at_start_and_end(self):
        """Test citations at the beginning and end of text."""
        text = "[1]This is a test sentence[2]"
        expected = "This is a test sentence"
        assert remove_citations(text) == expected

    def test_consecutive_citations(self):
        """Test consecutive citations."""
        text = "This is a test[1][2][3] sentence."
        expected = "This is a test sentence."
        assert remove_citations(text) == expected

    def test_malformed_brackets(self):
        """Test text with malformed brackets."""
        text = "This is a test[1 sentence with missing bracket."
        expected = "This is a test[1 sentence with missing bracket."
        assert remove_citations(text) == expected

    def test_only_citations(self):
        """Test string containing only citations."""
        text = "[1][2][3]"
        expected = ""
        assert remove_citations(text) == expected
