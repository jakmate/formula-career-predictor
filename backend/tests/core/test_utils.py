import pandas as pd
import numpy as np
from unittest.mock import patch

from app.core.utils import get_race_columns, extract_position, calculate_age


class TestGetRaceColumns:
    def test_identifies_race_columns_with_data(self):
        """Test identifying race columns with valid track codes and data."""
        df = pd.DataFrame({
            'Driver': ['Alice', 'Bob'],
            'BAH Sprint': [1, 2],
            'BAH Race': [2, 1],
            'SAU Sprint': [None, None],  # No data
            'MON Race': [3, 4],
            'Invalid': ['text', 'text'],  # Not a track code pattern
            'MON Sprint': [np.nan, np.nan]
        })

        result = get_race_columns(df)
        expected = ['BAH Sprint', 'BAH Race', 'MON Race']
        assert set(result) == set(expected)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = get_race_columns(df)
        assert result == []

    def test_no_race_columns(self):
        """Test DataFrame with no valid race columns."""
        df = pd.DataFrame({
            'Driver': ['Alice', 'Bob'],
            'Age': [25, 30],
            'Team': ['TeamA', 'TeamB']
        })

        result = get_race_columns(df)
        assert result == []

    def test_mixed_case_track_codes(self):
        """Test that only uppercase track codes are identified."""
        df = pd.DataFrame({
            'Driver': ['Alice', 'Bob'],
            'BAH Sprint': [1, 2],
            'bah race': [2, 1],  # lowercase
            'Mon Sprint': [3, 4]  # mixed case
        })

        result = get_race_columns(df)
        assert result == ['BAH Sprint']


class TestExtractPosition:
    def test_valid_positions(self):
        """Test extraction of valid numeric positions."""
        assert extract_position('1') == 1
        assert extract_position('10') == 10
        assert extract_position('1.0') == 1
        assert extract_position('5.5') == 5

    def test_invalid_inputs(self):
        """Test invalid inputs return None."""
        assert extract_position(None) is None
        assert extract_position('') is None
        assert extract_position('ABC') is None

    def test_edge_cases(self):
        """Test edge cases."""
        assert extract_position('0') == 0
        assert extract_position('-1') == -1
        assert extract_position('1.9') == 1


class TestCalculateAge:
    def test_valid_age_calculation(self):
        """Test valid age calculation."""
        df = pd.DataFrame({
            'dob': ['1990-05-15', '1985-12-01'],
            'year': [2020, 2025]
        })

        result = calculate_age(df)

        # 1990-05-15 to 2020-01-01 ≈ 29.6 years
        # 1985-12-01 to 2025-01-01 ≈ 34.1 years
        assert abs(result.loc[0, 'age'] - 29.6) < 0.1
        assert abs(result.loc[1, 'age'] - 39.1) < 0.1

    def test_missing_dob_column(self):
        """Test DataFrame without dob column."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'year': [2020, 2020]
        })

        result = calculate_age(df)
        assert 'age' in result.columns
        assert result['age'].isna().all()

    def test_invalid_dob_format(self):
        """Test invalid date of birth formats."""
        df = pd.DataFrame({
            'dob': ['invalid', '1990-13-40', '90-05-15', None],
            'year': [2020, 2020, 2020, 2020]
        })

        result = calculate_age(df)
        assert result['age'].isna().all()

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_age(df)
        assert result.empty

    def test_mixed_valid_invalid_data(self):
        """Test DataFrame with mix of valid and invalid data."""
        df = pd.DataFrame({
            'dob': ['1990-05-15', 'invalid', '1985-12-01', None],
            'year': [2020, 2020, 2020, 2020]
        })

        result = calculate_age(df)
        assert not pd.isna(result.loc[0, 'age'])  # Valid
        assert pd.isna(result.loc[1, 'age'])      # Invalid
        assert not pd.isna(result.loc[2, 'age'])  # Valid
        assert pd.isna(result.loc[3, 'age'])      # None

    @patch('builtins.print')
    @patch('app.core.utils.datetime')
    def test_exception_handling(self, mock_datetime, mock_print):
        """Test that exceptions are caught and handled."""
        # Make datetime.strptime raise an exception
        mock_datetime.strptime.side_effect = Exception("Test exception")
        mock_datetime.return_value = mock_datetime

        df = pd.DataFrame({
            'dob': ['1990-05-15'],
            'year': [2020]
        })

        result = calculate_age(df)
        assert 'age' in result.columns
        assert result['age'].isna().all()
        mock_print.assert_called()  # Error should be printed

    def test_short_dob_strings(self):
        """Test that short dob strings are handled."""
        df = pd.DataFrame({
            'dob': ['1990', '90-05', ''],
            'year': [2020, 2020, 2020]
        })

        result = calculate_age(df)
        assert result['age'].isna().all()
