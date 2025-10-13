import numpy as np
from datetime import datetime

from app.config import NOT_PARTICIPATED_CODES


def get_race_columns(df):
    """Identify race result columns based on track code patterns and data presence."""
    # Only consider columns that have non-null data in this DataFrame subset
    columns_with_data = []
    for col in df.columns:
        if df[col].notna().any():  # Column has at least one non-null value
            columns_with_data.append(col)

    track_codes = set()
    for col in columns_with_data:
        parts = col.split()
        if parts and len(parts[0]) >= 3:
            code_candidate = parts[0][:3]
            if code_candidate.isalpha() and code_candidate.isupper():
                track_codes.add(code_candidate)

    race_columns = []
    for col in columns_with_data:
        parts = col.split()
        if parts and len(parts[0]) >= 3:
            code_candidate = parts[0][:3]
            if code_candidate in track_codes:
                race_columns.append(col)

    return race_columns


def extract_position(result_str):
    """Extract numeric position from result string."""
    if not result_str or result_str in NOT_PARTICIPATED_CODES:
        return None

    try:
        clean_str = result_str.split()[0].replace('†', '')
        return int(float(clean_str))
    except (ValueError, IndexError):
        return None


def calculate_age(df):
    if df.empty:
        return df

    try:
        if 'dob' not in df.columns:
            df['age'] = np.nan
            return df

        ages = []
        for _, row in df.iterrows():
            try:
                if len(str(row['dob'])) != 10:
                    ages.append(np.nan)
                    continue

                dob = datetime.strptime(str(row['dob']), '%Y-%m-%d')
                season_start = datetime(int(row['year']), 1, 1)
                age = round((season_start - dob).days / 365.25, 1)
                ages.append(age)
            except (ValueError, TypeError):
                ages.append(np.nan)

        df['age'] = ages
        return df

    except Exception as e:
        print(f"Error in calculate_age: {e}")
        df['age'] = np.nan
        return df
