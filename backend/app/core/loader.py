import glob
import json
import os
import re
import pandas as pd

from app.config import DATA_DIR, PROFILES_DIR


FILE_PATTERNS = {
    'drivers': '{series}_{year}_drivers_standings.csv',
    'entries': '{series}_{year}_entries.csv',
    'teams': '{series}_{year}_teams_standings.csv',
    'qualifying': '{series}_{year}_qualifying_round_{round}.csv'
}


def get_file_pattern(file_type, series, year, round_num=None):
    """Get file pattern based on series type."""
    pattern = FILE_PATTERNS[file_type]
    if round_num:
        return pattern.format(series=series.lower(), year=year, round=round_num)
    return pattern.format(series=series.lower(), year=year)


def get_series_directories(series):
    """Get all directories for a series and their patterns."""
    directories = []

    series = os.path.join(DATA_DIR, series, "*")
    series_dirs = glob.glob(series)
    for year_dir in series_dirs:
        directories.append(year_dir)

    return directories


def load_all_entries_data(series):
    """Load all entries data for a series at once."""
    all_entries = []
    directories = get_series_directories(series)

    for year_dir in directories:
        year = os.path.basename(year_dir)
        try:
            year_int = int(year)
            entries_file = os.path.join(
                year_dir,
                get_file_pattern('entries', series, year)
            )

            if not os.path.exists(entries_file):
                return None
            entries_df = pd.read_csv(entries_file)

            if entries_df is not None:
                entries_df['year'] = year_int
                entries_df['series'] = series
                all_entries.append(entries_df)

        except Exception as e:
            print(f"Error loading entries for {year_dir}: {e}")

    return pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()


def load_year_data(year_dir, series, data_type):
    """Load data for a specific year and series."""
    year = os.path.basename(year_dir)

    try:
        year_int = int(year)

        # Get file paths
        data_file = os.path.join(
            year_dir,
            get_file_pattern(data_type, series, year)
        )

        if not os.path.exists(data_file):
            return None

        df = pd.read_csv(data_file)
        # Case of Konstantin Tereshchenko
        if data_type == 'drivers' and 'Pos' in df.columns:
            df = df.dropna(subset=['Pos'])

        df['year'] = year_int
        df['series'] = series

        return df

    except Exception as e:
        print(f"Error processing {year_dir}: {e}")
        return None


def load_standings_data(series, data_type):
    """Load standings data (drivers or teams) for a racing series."""
    all_data = []
    directories = get_series_directories(series)

    for year_dir in directories:
        df = load_year_data(year_dir, series, data_type)
        if df is not None:
            all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def load_qualifying_data(series):
    """Load qualifying data for a racing series across years."""
    all_qualifying_data = []
    directories = get_series_directories(series)

    for year_dir in directories:
        year = os.path.basename(year_dir)
        try:
            year_int = int(year)
            qualifying_dir = os.path.join(year_dir, 'qualifying')
            if not os.path.exists(qualifying_dir):
                continue

            # Load all qualifying round files
            qualifying_files = glob.glob(os.path.join(
                qualifying_dir,
                f"{series.lower()}_{year}_qualifying_round_*.csv")
            )

            year_qualifying_data = []
            for quali_file in qualifying_files:
                try:
                    df = pd.read_csv(quali_file)
                    df['year'] = year_int
                    df['series'] = series
                    df['round'] = os.path.basename(
                        quali_file).split('_')[-1].replace('.csv', '')
                    year_qualifying_data.append(df)
                except Exception as e:
                    print(f"Error loading quali file {quali_file}: {e}")

            if year_qualifying_data:
                all_qualifying_data.extend(year_qualifying_data)
        except Exception as e:
            print(f"Error processing quali data for {year_dir}: {e}")
            continue

    if all_qualifying_data:
        df = pd.concat(all_qualifying_data, ignore_index=True)
        return df
    return pd.DataFrame()


def get_driver_filename(driver_name):
    safe_name = re.sub(r'[^\w\s-]', '', driver_name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return f"{safe_name.lower()}.json"


def load_driver_data(df):
    """Add features from cached JSON profiles."""
    default_profile = {'dob': None, 'nationality': None}

    # Load cached profiles
    profiles = {}
    if os.path.exists(PROFILES_DIR):
        for driver in df['Driver'].unique():
            profile_file = os.path.join(PROFILES_DIR, get_driver_filename(driver))
            try:
                if os.path.exists(profile_file):
                    with open(profile_file, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)

                        # Check if driver was successfully scraped
                        if profile_data.get('scraped', True):
                            profiles[driver] = profile_data
                        else:
                            profiles[driver] = default_profile
                else:
                    profiles[driver] = default_profile
            except BaseException:
                profiles[driver] = default_profile

    # Add features
    feature_data = []
    for _, row in df.iterrows():
        profile = profiles.get(row['Driver'], {})
        feature_data.append({
            'dob': profile.get('dob'),
            'nationality': profile.get('nationality', 'Unknown')
        })

    for key in ['dob', 'nationality']:
        df[key] = [item[key] for item in feature_data]
    return df


def merge_entries(driver_df, entries_df):
    """Merge all entries data with driver standings at once."""
    if entries_df.empty:
        return driver_df

    # Process entries data to create team assignments
    all_team_data = []

    for (year, series), year_entries in entries_df.groupby(['year', 'series']):
        team_data = process_year_entries(year_entries)
        if not team_data.empty:
            team_data['year'] = year
            team_data['series'] = series
            all_team_data.append(team_data)

    if not all_team_data:
        return driver_df

    all_team_df = pd.concat(all_team_data, ignore_index=True)

    return driver_df.merge(
        all_team_df[['Driver', 'Team', 'team_count', 'year', 'series']],
        on=['Driver', 'year', 'series'],
        how='left'
    )


def process_year_entries(entries_df):
    """Process entries for a single year to determine team assignments."""
    if not all(col in entries_df.columns for col in ['Driver', 'Team']):
        return pd.DataFrame()

    multi_team_data = []

    for driver in entries_df['Driver'].unique():
        driver_entries = entries_df[entries_df['Driver'] == driver]

        if len(driver_entries) == 1:
            multi_team_data.append({
                'Driver': driver,
                'Team': driver_entries.iloc[0]['Team'],
                'team_count': 1
            })
        else:
            # Multi-team logic
            team_rounds = []
            for _, entry in driver_entries.iterrows():
                team = entry['Team']
                rounds_str = entry.get('Rounds', 'All')

                if rounds_str == 'All':
                    team_rounds.append((team, 999))
                else:
                    round_count = 0
                    if '–' in rounds_str or '-' in rounds_str:
                        parts = rounds_str.replace('–', '-').split(',')
                        for part in parts:
                            part = part.strip()
                            if '-' in part:
                                start, end = map(int, part.split('-'))
                                round_count += (end - start + 1)
                            else:
                                round_count += 1
                    else:
                        round_count = len(rounds_str.split(','))
                    team_rounds.append((team, round_count))

            primary_team = max(team_rounds, key=lambda x: x[1])[0]
            multi_team_data.append({
                'Driver': driver,
                'Team': primary_team,
                'team_count': len(driver_entries)
            })

    return pd.DataFrame(multi_team_data)


def calculate_position_percentile(team_df):
    """Calculate team position percentile."""
    team_metrics = []
    for year, year_data in team_df.groupby('year'):
        # Get unique teams and their positions
        team_positions = year_data.groupby('Team').agg({
            'Pos': 'first',
            'Points': 'first'
        }).reset_index()

        # Calculate relative performance metrics
        total_teams = len(team_positions)
        team_positions['team_pos_per'] = \
            (total_teams - team_positions['Pos'].astype(str).str.extract(
                r'(\d+)').astype(float) + 1) / total_teams

        # Add year and series info
        team_positions['year'] = year
        team_metrics.append(team_positions)

    return pd.concat(team_metrics, ignore_index=True) if team_metrics else pd.DataFrame()


def merge_team_data(driver_df, team_df):
    team_df = calculate_position_percentile(team_df)

    # For multi-team drivers, adjust team strength impact
    if 'team_count' in team_df.columns:
        multi_team_mask = driver_df['team_count'] > 1
        # Moderate the team performance impact for multi-team drivers
        team_df.loc[multi_team_mask, 'team_pos_per'] = \
            team_df.loc[multi_team_mask, 'team_pos_per'] * 0.8 + 0.1

    # Merge with driver data
    enhanced_df = driver_df.merge(
        team_df[['Team', 'year', 'Pos', 'team_pos_per', 'Points']],
        on=['Team', 'year'],
        how='left',
        suffixes=('', '_team')
    ).rename(columns={'Points_team': 'team_points', 'Pos_team': 'team_pos'})

    return enhanced_df


def load_data(series):
    print(f"Loading {series} driver data")
    driver_df = load_standings_data(series, 'drivers')

    print(f"Loading {series} team data")
    team_df = load_standings_data(series, 'teams')

    print(f"Loading {series} entries")
    entries_df = load_all_entries_data(series)

    print("Merging entries")
    df = merge_entries(driver_df, entries_df)

    print("Merge team data")
    df = merge_team_data(df, team_df)

    print("Loading driver data")
    df = load_driver_data(df)

    return df


if __name__ == "__main__":  # pragma: no cover
    load_data('F3')
