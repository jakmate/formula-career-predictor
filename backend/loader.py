import glob
import os
import pandas as pd


SERIES_CONFIG = {
    'F3': {'patterns': ['F3'], 'main_type': 'F3_Main'},  # w/o 'F3_European',
    'F2': {'patterns': ['F2'], 'main_type': 'F2_Main'},
    'F1': {'patterns': ['F1'], 'main_type': 'F1_Main'}
}

FILE_PATTERNS = {
    # 'F3_European': {
    #    'drivers': 'f3_euro_{year}_drivers_standings.csv',
    #    'entries': 'f3_euro_{year}_entries.csv',
    #    'teams': 'f3_euro_{year}_teams_standings.csv'
    # },
    'default': {
        'drivers': '{series}_{year}_drivers_standings.csv',
        'entries': '{series}_{year}_entries.csv',
        'teams': '{series}_{year}_teams_standings.csv',
        'qualifying': '{series}_{year}_qualifying_round_{round}.csv'
    }
}

COLUMN_MAPPING = {
    'Driver name': 'Driver',
    'Drivers': 'Driver',
    'Entrant': 'Team',
    'Teams': 'Team'
}


def clean_string_columns(df, columns):
    """Clean string columns by stripping whitespace."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def apply_column_mapping(df):
    """Apply standard column name mappings."""
    return df.rename(columns=COLUMN_MAPPING)


def get_file_pattern(series_type, file_type, series, year, round_num=None):
    """Get file pattern based on series type."""
    if series_type == 'F3_European':
        FILE_PATTERNS['F3_European'][file_type].format(year=year)

    pattern = FILE_PATTERNS['default'][file_type]
    if round_num:
        return pattern.format(series=series.lower(), year=year, round=round_num)
    return pattern.format(series=series.lower(), year=year)


def get_series_directories(series):
    """Get all directories for a series and their patterns."""
    config = SERIES_CONFIG[series]
    directories = []

    for series_pattern in config['patterns']:
        pattern = os.path.join("data", series_pattern, "*")
        series_dirs = glob.glob(pattern)
        for year_dir in series_dirs:
            directories.append((year_dir, series_pattern))

    return directories


def determine_series_type(series_pattern, series):
    """Determine the series type based on pattern."""
    if series_pattern == 'F3_European':
        return 'F3_European'
    return SERIES_CONFIG[series]['main_type']


def merge_entries_data(df, entries_file):
    """Merge entries data with driver standings."""
    if not os.path.exists(entries_file):
        return df

    entries_df = pd.read_csv(entries_file)

    # Remove columns from F3 European entries
    columns_to_drop = ['Chassis', 'Engine', 'Status']
    for col in columns_to_drop:
        if col in entries_df.columns:
            entries_df = entries_df.drop(columns=col)

    # Apply column mapping and clean data
    entries_df = apply_column_mapping(entries_df)
    entries_df = clean_string_columns(entries_df, ['Driver'])
    df = clean_string_columns(df, ['Driver'])

    entries_df['Driver'] = entries_df['Driver'].replace('Robert Visoiu', 'Robert Vișoiu')
    # Merge team data if both Driver and Team columns exist
    if all(col in entries_df.columns for col in ['Driver', 'Team']):
        # Create weighted team assignment for multi-team drivers
        multi_team_data = []

        for driver in entries_df['Driver'].unique():
            driver_entries = entries_df[entries_df['Driver'] == driver]

            if len(driver_entries) == 1:
                # Single team driver
                multi_team_data.append({
                    'Driver': driver,
                    'Team': driver_entries.iloc[0]['Team'],
                    'primary_team': driver_entries.iloc[0]['Team'],
                    'team_count': 1
                })
            else:
                # Multi-team driver - parse rounds to determine primary team
                team_rounds = []

                for _, entry in driver_entries.iterrows():
                    team = entry['Team']
                    rounds_str = entry.get('Rounds', 'All')

                    if rounds_str == 'All':
                        # Assign high weight for full season
                        team_rounds.append((team, 999))
                    else:
                        # Count individual rounds
                        round_count = 0
                        if '–' in rounds_str or '-' in rounds_str:
                            # Range like "1-5" or "7-8"
                            parts = rounds_str.replace('–', '-').split(',')
                            for part in parts:
                                part = part.strip()
                                if '-' in part:
                                    start, end = map(int, part.split('-'))
                                    round_count += (end - start + 1)
                                else:
                                    round_count += 1
                        else:
                            # Single round or comma-separated
                            round_count = len(rounds_str.split(','))

                        team_rounds.append((team, round_count))

                # Primary team is the one with most rounds
                primary_team = max(team_rounds, key=lambda x: x[1])[0]

                multi_team_data.append({
                    'Driver': driver,
                    'Team': primary_team,  # Use primary team for main Team column
                    'team_count': len(driver_entries)
                })

        team_data = pd.DataFrame(multi_team_data)
        df = df.merge(team_data[['Driver', 'Team', 'team_count']], on='Driver', how='left')

    return df


def load_year_data(year_dir, series_pattern, series, data_type):
    """Load data for a specific year and series."""
    year = os.path.basename(year_dir)

    try:
        year_int = int(year)
        series_type = determine_series_type(series_pattern, series)

        # Get file paths
        data_file = os.path.join(
            year_dir,
            get_file_pattern(series_type, data_type, series, year)
        )

        if not os.path.exists(data_file):
            return None

        df = pd.read_csv(data_file)
        # Case of Konstantin Tereshchenko
        if data_type == 'drivers' and 'Pos' in df.columns:
            df = df.dropna(subset=['Pos'])

        # Add metadata
        df['year'] = year_int
        df['series'] = series
        df['series_type'] = series_type

        # Special processing for driver data
        if data_type == 'drivers':
            # Merge entries data if available
            entries_file = os.path.join(
                year_dir,
                get_file_pattern(series_type, 'entries', series, year)
            )
            df = merge_entries_data(df, entries_file)

        return df

    except Exception as e:
        print(f"Error processing {year_dir}: {e}")
        return None


def load_standings_data(series, data_type):
    """Load standings data (drivers or teams) for a racing series."""
    all_data = []
    directories = get_series_directories(series)

    for year_dir, series_pattern in directories:
        df = load_year_data(year_dir, series_pattern, series, data_type)
        if df is not None:
            all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def load_qualifying_data(series='F3'):
    """Load qualifying data for a racing series across years."""
    all_qualifying_data = []
    directories = get_series_directories(series)

    for year_dir, series_pattern in directories:
        # Skip F3_European as it doesn't have qualifying data
        if series_pattern == 'F3_European':
            continue

        year = os.path.basename(year_dir)
        try:
            year_int = int(year)
            # Skip 2011 as it doesn't have qualifying data
            if year_int == 2011:
                continue

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
        return pd.concat(all_qualifying_data, ignore_index=True)
    return pd.DataFrame()


def calculate_position_percentile(team_df):
    """Calculate team position percentile."""
    team_metrics = []
    for (year, series_type), year_data in team_df.groupby(['year', 'series_type']):
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
        team_positions['series_type'] = series_type

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
        team_df[['Team', 'year', 'series_type', 'Pos', 'team_pos_per', 'Points']],
        on=['Team', 'year', 'series_type'],
        how='left',
        suffixes=('', '_team')
    ).rename(columns={'Points_team': 'team_points', 'Pos_team': 'team_pos'})

    return enhanced_df


def load_all_data():
    print("Loading F3 data...")
    f3_df = load_standings_data('F3', 'drivers')

    print("Loading F2 data...")
    f2_df = load_standings_data('F2', 'drivers')

    print("Loading F3 team championship data...")
    f3_team_df = load_standings_data('F3', 'teams')

    print("Merge team data")
    f3_df = merge_team_data(f3_df, f3_team_df)

    # print("Rows with NaN values:")
    # print(f3_qualifying_df.isna().sum())
    # f3_qualifying_df.isna().sum().to_csv('nan_counts.csv')
    # print(f3_df[f3_df['Team'].isna()])
    # print(f3_df[f3_df['year'] == 2012])

    return f2_df, f3_df


if __name__ == "__main__":
    f2_df, f3_df = load_all_data()
