import glob
import json
import os
import random
import numpy as np
import optuna
import pandas as pd
import re
import tensorflow as tf
import torch.optim as optim
import torch.nn as nn
import torch
import xgboost as xgb
from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 69
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SERIES_CONFIG = {
    'F3': {'patterns': ['F3', 'F3_European'], 'main_type': 'F3_Main'},
    'F2': {'patterns': ['F2'], 'main_type': 'F2_Main'},
    'F1': {'patterns': ['F1'], 'main_type': 'F1_Main'}
}

FILE_PATTERNS = {
    'F3_European': {
        'drivers': 'f3_euro_{year}_drivers_standings.csv',
        'entries': 'f3_euro_{year}_entries.csv',
        'teams': 'f3_euro_{year}_teams_standings.csv'
    },
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
    'Entrant': 'Team'
}

NOT_PARTICIPATED_CODES = ['', 'DNS', 'WD', 'DNQ', 'DNA', 'C', 'EX']
RETIREMENT_CODES = ['Ret', 'NC', 'DSQ']


def clean_string_columns(df, columns):
    """Clean string columns by stripping whitespace."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def apply_column_mapping(df):
    """Apply standard column name mappings."""
    return df.rename(columns=COLUMN_MAPPING)


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
        if parts:
            if len(parts[0]) >= 3:
                code_candidate = parts[0][:3]
                if code_candidate.isalpha() and code_candidate.isupper():
                    track_codes.add(code_candidate)

    race_columns = []
    for col in columns_with_data:
        parts = col.split()
        if parts:
            if len(parts[0]) >= 3:
                code_candidate = parts[0][:3]
                if code_candidate in track_codes:
                    race_columns.append(col)

    return race_columns


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
                    'primary_team': primary_team,
                    'team_count': len(driver_entries)
                })

        team_data = pd.DataFrame(multi_team_data)
        df = df.merge(team_data[['Driver', 'Team', 'team_count']], on='Driver', how='left')
        df['team_count'] = df['team_count'].fillna(1)

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


def create_target_variable(f3_df, f2_df):
    """Create target variable for F2 participation after last F3 season."""
    if f3_df.empty or f2_df.empty:
        f3_df['moved_to_f2'] = np.nan
        return f3_df

    # Initialize target column
    f3_df['moved_to_f2'] = 0
    max_f2_year = f2_df['year'].max()

    # Get last F3 season for each driver
    last_f3_seasons = f3_df.groupby('Driver')['year'].max().reset_index()

    # Process F2 data to determine participation
    f2_participation = []
    for year, year_df in f2_df.groupby('year'):
        race_cols = get_race_columns(year_df)
        total_races = len(race_cols)
        if total_races == 0:
            continue

        for _, row in year_df.iterrows():
            driver = row['Driver']
            participation = 0
            for col in race_cols:
                result = str(row[col]).strip()
                if not result or result in NOT_PARTICIPATED_CODES:
                    continue
                participation += 1

            # For 2025, count any participation (>0 races)
            # For other years, use 50% threshold
            threshold = 0 if year == 2025 else total_races * 0.5

            f2_participation.append({
                'driver': driver,
                'year': year,
                'participated': participation > threshold
            })

    f2_participation_df = pd.DataFrame(f2_participation)

    # Determine target values
    moved_drivers = {}
    for _, row in last_f3_seasons.iterrows():
        driver = row['Driver']
        last_f3_year = row['year']

        # Skip if we can't observe future F2 seasons
        if last_f3_year + 1 > max_f2_year:
            moved_drivers[(driver, last_f3_year)] = np.nan
            continue

        # Check next years for F2 participation
        moved = 0
        for offset in [1]:
            target_year = last_f3_year + offset
            if target_year > max_f2_year:
                break

            participation = f2_participation_df[
                (f2_participation_df['driver'] == driver) &
                (f2_participation_df['year'] == target_year)
            ]

            if not participation.empty and participation['participated'].iloc[0]:
                moved = 1
                break

        moved_drivers[(driver, last_f3_year)] = moved

    # Apply target values
    for idx, row in f3_df.iterrows():
        driver = row['Driver']
        year = row['year']
        if (driver, year) in moved_drivers:
            f3_df.at[idx, 'moved_to_f2'] = moved_drivers[(driver, year)]

    return f3_df


def calculate_team_performance_metrics(team_df):
    """Calculate team performance metrics from standings."""
    if team_df.empty:
        return pd.DataFrame()

    team_metrics = []

    for (year, series_type), year_data in team_df.groupby(['year', 'series_type']):
        # Get unique teams and their positions
        team_positions = year_data.groupby('Team')['Pos'].first().reset_index()

        # Convert position to numeric, handling ties
        team_positions['team_pos'] = team_positions['Pos'].astype(
            str).str.extract(r'(\d+)').astype(float)

        # Calculate team points
        team_points = year_data.groupby('Team')['Points'].first().reset_index()
        team_positions = team_positions.merge(team_points, on='Team', how='left')

        # Calculate relative performance metrics
        total_teams = len(team_positions)
        team_positions['team_pos_per'] = \
            (total_teams - team_positions['team_pos'] + 1) / total_teams

        # Add year and series info
        team_positions['year'] = year
        team_positions['series_type'] = series_type

        team_metrics.append(team_positions)

    return pd.concat(team_metrics, ignore_index=True) if team_metrics else pd.DataFrame()


def enhance_with_team_data(driver_df, team_df):
    """Add team championship metrics to driver data."""
    if team_df.empty:
        # Add default values if no team data
        driver_df['team_pos'] = np.nan
        driver_df['team_pos_per'] = 0.5
        driver_df['team_points'] = 0
        return driver_df

    # Calculate team metrics
    team_metrics = calculate_team_performance_metrics(team_df)

    if team_metrics.empty:
        driver_df['team_pos'] = np.nan
        driver_df['team_pos_per'] = 0.5
        driver_df['team_points'] = 0
        return driver_df

    # Handle missing series_type in driver data
    if 'series_type' not in driver_df.columns:
        driver_df['series_type'] = 'F3_Main'

    # Merge with driver data
    merge_cols = ['Team', 'year', 'series_type']
    team_feature_cols = ['team_pos', 'team_pos_per', 'Points']

    enhanced_df = driver_df.merge(
        team_metrics[merge_cols + team_feature_cols],
        on=merge_cols,
        how='left',
        suffixes=('', '_team')
    )

    # Rename team points column to avoid confusion
    enhanced_df = enhanced_df.rename(columns={'Points_team': 'team_points'})

    # For multi-team drivers, adjust team strength impact
    if 'team_count' in enhanced_df.columns:
        multi_team_mask = enhanced_df['team_count'] > 1
        # Moderate the team performance impact for multi-team drivers
        enhanced_df.loc[multi_team_mask, 'team_pos_per'] = \
            enhanced_df.loc[multi_team_mask, 'team_pos_per'] * 0.8 + 0.2 * 0.5

    # Fill missing values
    enhanced_df['team_pos'] = enhanced_df['team_pos'].fillna(enhanced_df['team_pos'].median())
    enhanced_df['team_pos_per'] = enhanced_df['team_pos_per'].fillna(0.5)
    enhanced_df['team_points'] = enhanced_df['team_points'].fillna(0)

    return enhanced_df


def calculate_teammate_performance(df):
    """Calculate performance metrics relative to teammates."""
    if 'Team' not in df.columns:
        return df

    team_performance = []

    # Group by year and team to find teammates
    for (year, team), team_df in df.groupby(['year', 'Team']):
        if len(team_df) < 2:  # Skip teams with only one driver
            continue

        race_cols = get_race_columns(team_df)
        if not race_cols:
            continue

        for _, driver_row in team_df.iterrows():
            driver = driver_row['Driver']
            is_multi_team = driver_row.get('team_count', 1) > 1

            # Get teammates (excluding current driver)
            teammates_df = team_df[team_df['Driver'] != driver]

            if teammates_df.empty:
                continue

            # Calculate head-to-head record
            h2h_wins = 0
            h2h_total = 0
            better_finishes = 0

            for race_col in race_cols:
                driver_result = str(driver_row[race_col]).strip()
                if not driver_result or driver_result in NOT_PARTICIPATED_CODES:
                    continue

                try:
                    driver_pos = int(
                        driver_result.split()[0].replace(
                            '†', '').replace('Ret', '999'))
                except BaseException:
                    continue

                # Compare with each teammate
                for _, teammate_row in teammates_df.iterrows():
                    teammate_result = str(teammate_row[race_col]).strip()
                    if not teammate_result or teammate_result in NOT_PARTICIPATED_CODES:
                        continue

                    try:
                        teammate_pos = int(
                            teammate_result.split()[0].replace(
                                '†', '').replace('Ret', '999'))

                        h2h_total += 1

                        if driver_pos < teammate_pos:  # Lower pos number = better finish
                            h2h_wins += 1
                            better_finishes += 1

                    except BaseException:
                        continue

            # Calculate average position difference
            pos_differences = []
            for race_col in race_cols:
                driver_result = str(driver_row[race_col]).strip()
                if not driver_result or driver_result in NOT_PARTICIPATED_CODES:
                    continue

                try:
                    driver_pos = int(
                        driver_result.split()[0].replace(
                            '†', '').replace('Ret', '25'))
                except BaseException:
                    continue

                teammate_positions = []
                for _, teammate_row in teammates_df.iterrows():
                    teammate_result = str(teammate_row[race_col]).strip()
                    if not teammate_result or teammate_result in NOT_PARTICIPATED_CODES:
                        continue

                    try:
                        teammate_pos = int(
                            teammate_result.split()[0].replace(
                                '†', '').replace('Ret', '25'))
                        teammate_positions.append(teammate_pos)
                    except BaseException:
                        continue

                if teammate_positions:
                    avg_teammate_pos = np.mean(teammate_positions)
                    pos_differences.append(
                        avg_teammate_pos - driver_pos)  # Positive = driver better

            avg_pos_difference = np.mean(
                pos_differences) if pos_differences else 0

            # Adjust teammate metrics for multi-team drivers (reduce weight)
            if is_multi_team and h2h_total > 0:
                avg_pos_difference *= 0.7  # Reduce impact for partial teammate comparisons
                h2h_total = int(h2h_total * 0.7)

            team_performance.append({
                'Driver': driver,
                'year': year,
                'Team': team,
                'avg_pos_vs_teammates': avg_pos_difference,
                'teammate_battles': h2h_total,
                'is_multi_team': is_multi_team
            })

    # Convert to DataFrame and merge with original
    team_perf_df = pd.DataFrame(team_performance)
    if not team_perf_df.empty:
        df = df.merge(team_perf_df[['Driver',
                                    'year',
                                    'avg_pos_vs_teammates',
                                    'teammate_battles',
                                    'is_multi_team']],
                      on=['Driver',
                          'year'],
                      how='left')

        # Fill NaN values for drivers without teammates
        df['avg_pos_vs_teammates'] = df['avg_pos_vs_teammates'].fillna(0)
        df['teammate_battles'] = df['teammate_battles'].fillna(0)
        df['is_multi_team'] = df['is_multi_team'].infer_objects(copy=False)
    else:
        df['avg_pos_vs_teammates'] = 0
        df['teammate_battles'] = 0
        df['is_multi_team'] = False

    return df


def add_driver_features(features_df, f3_df):
    """Add features from cached JSON profiles."""
    profiles_dir = "data/driver_profiles"

    def get_driver_filename(driver_name):
        safe_name = re.sub(r'[^\w\s-]', '', driver_name)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        return f"{safe_name.lower()}.json"

    def calculate_age(dob_str, competition_year):
        if not dob_str:
            return None
        try:
            if len(dob_str) == 10:
                dob = datetime.strptime(dob_str, '%Y-%m-%d')
                season_start = datetime(competition_year, 1, 1)
                age = (season_start - dob).days / 365.25
                return round(age, 1)
        except BaseException:
            return None

    # Load cached profiles
    profiles = {}
    if os.path.exists(profiles_dir):
        for driver in f3_df['Driver'].unique():
            profile_file = os.path.join(
                profiles_dir, get_driver_filename(driver))
            if os.path.exists(profile_file):
                try:
                    with open(profile_file, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)

                        # Check if driver was successfully scraped
                        if profile_data.get('scraped', True):
                            profiles[driver] = profile_data
                        else:
                            # Driver exists but wasn't scraped - treat as no data
                            profiles[driver] = {'dob': None, 'nationality': None}

                except BaseException:
                    profiles[driver] = {'dob': None, 'nationality': None}
            else:
                profiles[driver] = {'dob': None, 'nationality': None}

    # Add features
    dobs, ages, nationalities = [], [], []

    for _, row in features_df.iterrows():
        driver = row['driver']
        year = row['year']

        profile = profiles.get(driver, {})

        dob = profile.get('dob')
        dobs.append(dob)

        age = calculate_age(profile.get('dob'), year)
        ages.append(age)

        nationality = profile.get('nationality', 'Unknown')
        nationalities.append(nationality)

    features_df['dob'] = dobs
    features_df['age'] = ages
    features_df['nationality'] = nationalities

    median_age = features_df['age'].median()
    features_df['age'] = features_df['age'].fillna(median_age)

    return features_df


def calculate_qualifying_features(df, qualifying_df):
    """Calculate qualifying-based features for drivers."""
    if qualifying_df.empty:
        df['avg_quali_pos'] = np.nan
        df['std_quali_pos'] = np.nan
        return df

    # Apply column mapping to qualifying data
    for old_name, new_name in COLUMN_MAPPING.items():
        if old_name in qualifying_df.columns:
            qualifying_df = qualifying_df.rename(columns={old_name: new_name})

    # Clean driver names
    qualifying_df = clean_string_columns(qualifying_df, ['Driver'])

    # Calculate qualifying statistics for each driver-year combination
    qualifying_stats = []

    for (driver, year), driver_year_data in qualifying_df.groupby(['Driver', 'year']):
        qualifying_positions = []

        # Extract qualifying positions from all rounds
        for idx, row in driver_year_data.iterrows():
            # Define priority order for qualifying position columns
            position_columns = [
                'Grid',     # Highest priority
                'GridFR',  # Secondary priority
                'R2',       # Tertiary priority
                'Pos',      # Fallback options
                'Position'
            ]
            position_value = None

            # Check columns in priority order
            for col in position_columns:
                if col in row and pd.notna(row[col]):
                    raw_value = row[col]

                    # Handle numeric values directly
                    if isinstance(raw_value, (int, float)):
                        position_value = int(raw_value)
                        break

                    # Convert to string and clean
                    str_value = str(raw_value).strip()

                    # Skip non-participation codes
                    if str_value in NOT_PARTICIPATED_CODES:
                        position_value = None
                        break

                    # Extract numeric position using regex (max 2 digits)
                    match = re.search(r'\b\d{1,2}\b', str_value)
                    if match:
                        position_value = int(match.group())
                        break

            if position_value is not None:
                qualifying_positions.append(position_value)

        # Calculate statistics only if we have valid positions
        if qualifying_positions:
            avg_quali = np.mean(qualifying_positions)
            std_quali = np.std(qualifying_positions) if len(qualifying_positions) > 1 else 0
        else:
            avg_quali = np.nan
            std_quali = np.nan

        qualifying_stats.append({
            'Driver': driver,
            'year': year,
            'avg_quali_pos': avg_quali,
            'std_quali_pos': std_quali
        })

    # Convert to DataFrame and merge with main data
    quali_stats_df = pd.DataFrame(qualifying_stats)

    if not quali_stats_df.empty:
        df = df.merge(quali_stats_df[['Driver', 'year', 'avg_quali_pos', 'std_quali_pos']],
                      on=['Driver', 'year'],
                      how='left')
    else:
        df['avg_quali_pos'] = np.nan
        df['std_quali_pos'] = np.nan

    return df


def engineer_features(df):
    """Create features for ML models"""
    if df.empty:
        return pd.DataFrame()

    df = calculate_teammate_performance(df)
    df = df.sort_values(by=['Driver', 'year'])
    df['years_in_f3'] = df.groupby('Driver').cumcount()

    features_df = pd.DataFrame()
    features_df['year'] = df['year']
    features_df['driver'] = df['Driver']
    features_df['final_pos'] = df['Pos'].astype(str).str.extract(r'(\d+)').astype(float)
    features_df['points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
    features_df['years_in_f3'] = df['years_in_f3']
    features_df['series_type'] = df.get('series_type', 'Unknown')
    features_df['is_f3_european'] = (features_df['series_type'] == 'F3_European').astype(int)

    features_df['team'] = df.get('Team')
    features_df['team_pos'] = df.get('team_pos', np.nan)
    features_df['team_pos_per'] = df.get('team_pos_per', 0.5)
    features_df['team_points'] = df.get('team_points', 0)

    race_cols_cache = {}
    wins, podiums, points_finishes, dnfs, races_completed = [], [], [], [], []
    participation_rates, field_sizes = [], []
    avg_finish_positions = []
    std_finish_positions = []

    for _, row in df.iterrows():
        # Create cache key for this year/series combination
        cache_key = (row['year'], row.get('series_type', 'F3_Main'))

        if cache_key not in race_cols_cache:
            # Get all rows for this year/series to determine race columns
            year_series_data = df[
                (df['year'] == row['year']) &
                (df.get('series_type', 'F3_Main') == row.get('series_type', 'F3_Main'))
            ]
            race_cols_cache[cache_key] = get_race_columns(year_series_data)

        race_cols = race_cols_cache[cache_key]

        driver_wins = 0
        driver_podiums = 0
        driver_points = 0
        driver_dnfs = 0
        driver_races = 0
        total_scheduled_races = len(race_cols)

        # Calculate field size for this year/series
        year_series_data = df[
            (df['year'] == row['year']) &
            (df.get('series_type', 'F3_Main') == row.get('series_type', 'F3_Main'))
        ]
        field_size = len(year_series_data)
        field_sizes.append(field_size)

        finish_positions = []

        for col in race_cols:
            if col not in row or pd.isna(row[col]):
                continue

            result = str(row[col]).strip()
            if not result:
                continue

            if result not in NOT_PARTICIPATED_CODES:
                driver_races += 1

            if any(x in result for x in RETIREMENT_CODES):
                if result != 'NC':
                    driver_dnfs += 1
                continue

            try:
                pos = int(
                    result.split()[0].replace('†', '').replace(
                        'F', '').replace('P', ''))
                finish_positions.append(pos)

                if pos == 1:
                    driver_wins += 1
                    driver_podiums += 1
                    driver_points += 1
                elif pos <= 3:
                    driver_podiums += 1
                    driver_points += 1
                elif pos <= 10:
                    driver_points += 1
            except BaseException:
                continue

        if finish_positions:
            avg_position = np.mean(finish_positions)
            std_position = np.std(finish_positions)
        else:
            avg_position = np.nan
            std_position = np.nan

        avg_finish_positions.append(avg_position)
        std_finish_positions.append(std_position)

        if total_scheduled_races > 0:
            participation_rate = driver_races / total_scheduled_races
        else:
            participation_rate = 0

        wins.append(driver_wins)
        podiums.append(driver_podiums)
        points_finishes.append(driver_points)
        dnfs.append(driver_dnfs)
        races_completed.append(driver_races if driver_races > 0 else 1)
        participation_rates.append(participation_rate)

    features_df['wins'] = wins
    features_df['podiums'] = podiums
    features_df['points_finishes'] = points_finishes
    features_df['dnfs'] = dnfs
    features_df['races_completed'] = races_completed
    features_df['participation_rate'] = participation_rates
    features_df['field_size'] = field_sizes
    features_df['avg_finish_pos'] = avg_finish_positions
    features_df['std_finish_pos'] = std_finish_positions

    features_df['avg_pos_vs_teammates'] = df.get('avg_pos_vs_teammates', 0)
    features_df['teammate_battles'] = df.get('teammate_battles', 0)

    features_df['avg_quali_pos'] = df.get('avg_quali_pos', np.nan)
    features_df['std_quali_pos'] = df.get('std_quali_pos', np.nan)

    features_df = add_driver_features(features_df, f3_df)

    features_df['win_rate'] = features_df['wins'] / \
        features_df['races_completed']
    features_df['podium_rate'] = features_df['podiums'] / \
        features_df['races_completed']
    features_df['dnf_rate'] = features_df['dnfs'] / \
        features_df['races_completed']
    features_df['top_10_rate'] = features_df['points_finishes'] / \
        features_df['races_completed']
    features_df['points_vs_team_strength'] = features_df['points'] / \
        (features_df['team_points'] + 1)
    features_df['pos_vs_team_strength'] = features_df['final_pos'] * features_df['team_pos_per']

    return features_df


def create_tensorflow_dnn(input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden_layers[0], activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate)
    ])

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

class RacingPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(RacingPredictor, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_pytorch_model(X_train_scaled, y_train, X_val_scaled, y_val, input_dim, hidden_dims, dropout_rate, learning_rate, batch_size, epochs=100):
    """Train PyTorch model with proper training loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = RacingPredictor(input_dim, hidden_dims, dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def optimize_traditional_hyperparams(X_train, y_train, model_type='xgboost', n_trials=50):
    """Optimize hyperparameters using Optuna"""
    def objective(trial):
        if model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            model = xgb.XGBClassifier(random_state=SEED, **params)
            
        elif model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            }
            model = RandomForestClassifier(random_state=SEED, class_weight='balanced', **params)
            
        elif model_type == 'logistic':
            params = {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
            }
            model = LogisticRegression(
                random_state=SEED, penalty='elasticnet', solver='saga',
                class_weight='balanced', max_iter=10000, **params
            )
        
        elif model_type == 'mlp':
            # MLP hyperparameters
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_layer_sizes = []
            for i in range(n_layers):
                size = trial.suggest_int(f'layer_{i}_size', 32, 256)
                hidden_layer_sizes.append(size)
            
            params = {
                'hidden_layer_sizes': tuple(hidden_layer_sizes),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.2,
                'n_iter_no_change': 20
            }
            model = MLPClassifier(random_state=SEED, **params)

        # Use SMOTE + model pipeline
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='roc_auc')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

def optimize_deep_learning_hyperparams(X_train_scaled, y_train, model_type='keras', n_trials=30):
    """Optimize deep learning hyperparameters"""
    def objective(trial):
        if model_type == 'keras':
            # Suggest architecture
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_dims = []
            for i in range(n_layers):
                dim = trial.suggest_int(f'hidden_dim_{i}', 32, 256)
                hidden_dims.append(dim)
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            model = create_tensorflow_dnn(
                X_train_scaled.shape[1], hidden_dims, dropout_rate, learning_rate
            )
            
            # Quick training for evaluation
            early_stop = EarlyStopping(patience=5, restore_best_weights=True)
            model.fit(
                X_train_scaled, y_train, validation_split=0.2, epochs=20,
                batch_size=batch_size, callbacks=[early_stop], verbose=0
            )
            
            val_loss = min(model.history.history['val_loss'])
            return -val_loss  # Minimize loss = maximize negative loss
            
        elif model_type == 'pytorch':
            # PyTorch hyperparameters
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_dims = []
            for i in range(n_layers):
                dim = trial.suggest_int(f'hidden_dim_{i}', 32, 256)
                hidden_dims.append(dim)
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Split for validation
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=SEED, stratify=y_train
            )
            
            model = train_pytorch_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                X_train_scaled.shape[1], hidden_dims, dropout_rate, 
                learning_rate, batch_size, epochs=30
            )
            
            # Evaluate on validation set
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_fold).to(device)
                y_val_tensor = torch.FloatTensor(y_val_fold.values).to(device)
                outputs = model(X_val_tensor).squeeze()
                val_loss = nn.BCEWithLogitsLoss()(outputs, y_val_tensor).item()
            
            return -val_loss
            
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

def create_ensemble_model(traditional_models, X_train, y_train):
    """Create voting ensemble from best traditional models"""
    # Select top 3 models based on cross-validation performance
    model_scores = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    
    for name, pipeline in traditional_models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        model_scores[name] = scores.mean()
    
    # Get top 3 models
    top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    ensemble_models = [(name, traditional_models[name]) for name, _ in top_models]
    
    ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
    return ensemble

def train_models(df):
    """Enhanced training with PyTorch and MLP"""
    if df.empty:
        print("No data available for training")
        return {}, {}, None, None, None

    df_clean = df.dropna(subset=['moved_to_f2', 'final_pos', 'points'])
    feature_cols = [
        'final_pos', 'win_rate', 'podium_rate', 'dnf_rate', 'top_10_rate',
        'years_in_f3', 'age', 'avg_pos_vs_teammates', 'teammate_battles',
        'participation_rate', 'is_f3_european', 'team_pos', 'team_points',
        'points_vs_team_strength', 'pos_vs_team_strength',
        'avg_finish_pos', 'std_finish_pos', 'avg_quali_pos', 'std_quali_pos'
    ]

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['moved_to_f2']

    print(f"Dataset size: {len(X)} drivers")
    print(f"F2 progressions: {y.sum()} ({y.mean():.2%})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    traditional_results = {}
    deep_results = {}

    # Optimize traditional models (including MLP)
    print("\n" + "=" * 50)
    print("OPTIMIZING TRADITIONAL MODELS")
    print("=" * 50)

    model_types = ['xgboost', 'random_forest', 'logistic', 'mlp']
    for model_type in model_types:
        print(f"\nOptimizing {model_type}...")
        best_params = optimize_traditional_hyperparams(X_train, y_train, model_type)
        print(f"Best params: {best_params}")

        # Create optimized model
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(random_state=SEED, **best_params)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                random_state=SEED, class_weight='balanced', **best_params
            )
        elif model_type == 'logistic':
            model = LogisticRegression(
                random_state=SEED, penalty='elasticnet', solver='saga',
                class_weight='balanced', max_iter=10000, **best_params
            )
        elif model_type == 'mlp':
            mlp_params = {k: v for k, v in best_params.items() 
                         if k not in ['n_layers'] and not k.startswith('layer_')}
            model = MLPClassifier(random_state=SEED, **mlp_params)

        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        
        # Add calibration
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(y_pred_proba, y_test)
        pipeline.calibrator = iso_reg
        
        traditional_results[f'Optimized_{model_type}'] = pipeline

    # Create ensemble
    print("\nCreating ensemble model...")
    ensemble = create_ensemble_model(traditional_results, X_train, y_train)
    ensemble.fit(X_train, y_train)
    
    # Add calibration to ensemble
    y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
    iso_reg_ensemble = IsotonicRegression(out_of_bounds='clip')
    iso_reg_ensemble.fit(y_pred_proba_ensemble, y_test)
    ensemble.calibrator = iso_reg_ensemble
    
    traditional_results['Ensemble'] = ensemble

    # Optimize deep learning
    print("\n" + "=" * 50)
    print("OPTIMIZING DEEP LEARNING MODELS")
    print("=" * 50)

    # Optimize Keras
    print("Optimizing Keras DNN...")
    best_keras_params = optimize_deep_learning_hyperparams(X_train_scaled, y_train, 'keras')
    print(f"Best Keras params: {best_keras_params}")

    # Train optimized Keras model
    hidden_dims = []
    for i in range(best_keras_params['n_layers']):
        if f'hidden_dim_{i}' in best_keras_params:
            hidden_dims.append(best_keras_params[f'hidden_dim_{i}'])

    dnn_model = create_tensorflow_dnn(
        X_train_scaled.shape[1], 
        hidden_dims, 
        best_keras_params['dropout_rate'],
        best_keras_params['learning_rate']
    )

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)
    ]

    dnn_model.fit(
        X_train_scaled, y_train, validation_split=0.2, epochs=150,
        batch_size=best_keras_params['batch_size'], callbacks=callbacks, verbose=0
    )

    # Add calibration to Keras model
    probas_dnn = dnn_model.predict(X_test_scaled, verbose=0).flatten()
    iso_reg_dnn = IsotonicRegression(out_of_bounds='clip')
    iso_reg_dnn.fit(probas_dnn, y_test)
    dnn_model.calibrator = iso_reg_dnn

    deep_results['Optimized_Keras_DNN'] = dnn_model

    # Optimize PyTorch
    print("Optimizing PyTorch DNN...")
    best_pytorch_params = optimize_deep_learning_hyperparams(X_train_scaled, y_train, 'pytorch')
    print(f"Best PyTorch params: {best_pytorch_params}")

    # Train optimized PyTorch model
    pytorch_hidden_dims = []
    for i in range(best_pytorch_params['n_layers']):
        if f'hidden_dim_{i}' in best_pytorch_params:
            pytorch_hidden_dims.append(best_pytorch_params[f'hidden_dim_{i}'])

    # Split for validation
    X_train_pt, X_val_pt, y_train_pt, y_val_pt = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    pytorch_model = train_pytorch_model(
        X_train_pt, y_train_pt, X_val_pt, y_val_pt,
        X_train_scaled.shape[1], pytorch_hidden_dims,
        best_pytorch_params['dropout_rate'],
        best_pytorch_params['learning_rate'],
        best_pytorch_params['batch_size'],
        epochs=150
    )

    # Add calibration to PyTorch model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pytorch_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        pytorch_probas = torch.sigmoid(pytorch_model(X_test_tensor)).cpu().numpy().flatten()
    
    iso_reg_pytorch = IsotonicRegression(out_of_bounds='clip')
    iso_reg_pytorch.fit(pytorch_probas, y_test)
    pytorch_model.calibrator = iso_reg_pytorch

    deep_results['Optimized_PyTorch_DNN'] = pytorch_model

    # Evaluate all models
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    for name, model in traditional_results.items():
        y_pred = model.predict(X_test)
        print(f"\n{name}:")
        print(classification_report(y_test, y_pred))

    # Evaluate deep learning models
    probas_dnn = dnn_model.predict(X_test_scaled, verbose=0).flatten()
    pred_dnn = (probas_dnn > 0.5).astype(int)
    print(f"\nOptimized Keras DNN:")
    print(classification_report(y_test, pred_dnn))

    pred_pytorch = (pytorch_probas > 0.5).astype(int)
    print(f"\nOptimized PyTorch DNN:")
    print(classification_report(y_test, pred_pytorch))

    return traditional_results, deep_results, X_test, y_test, feature_cols, scaler


def predict_drivers(all_models, df, feature_cols, scaler=None):
    """Make predictions for F3 2025 drivers"""
    current_year = 2025
    current_df = df[df['year'] == current_year].copy()
    if current_df.empty:
        current_year = df['year'].max()
        current_df = df[df['year'] == current_year].copy()
    if current_df.empty:
        print("No current data found for predictions")
        return pd.DataFrame()

    X_current = current_df[feature_cols].fillna(0)
    results = None

    for model_type, models in all_models.items():
        print(f"\n{model_type} Predictions:")
        print("=" * 70)

        # Scale features for deep learning models
        if model_type == 'Deep Learning' and scaler is not None:
            X_processed = scaler.transform(X_current)
        else:
            X_processed = X_current

        for name, model in models.items():
            try:
                # Get raw probabilities based on model type
                if model_type == 'Deep Learning':
                    if 'Keras' in name:  # Keras model
                        raw_probas = model.predict(X_processed, verbose=0).flatten()
                    elif 'PyTorch' in name:  # PyTorch model
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.eval()
                        with torch.no_grad():
                            X_torch = torch.FloatTensor(X_processed).to(device)
                            logits = model(X_torch)
                            raw_probas = torch.sigmoid(logits).cpu().numpy().flatten()
                else:  # Traditional models
                    raw_probas = model.predict_proba(X_processed)[:, 1]

                # Apply calibration if available
                if hasattr(model, 'calibrator') and model.calibrator is not None:
                    calibrated_probas = model.calibrator.transform(raw_probas)
                else:
                    calibrated_probas = raw_probas
                    
                empirical_pct = calibrated_probas * 100.0

                # Create results DataFrame
                results = pd.DataFrame({
                    'Driver': current_df['driver'],
                    'Nat.': current_df['nationality'],
                    'Pos': current_df['final_pos'],
                    'Avg Pos': current_df['avg_finish_pos'],
                    'Std Pos': current_df['std_finish_pos'],
                    'Avg Quali': current_df['avg_quali_pos'],
                    'Std Quali': current_df['std_quali_pos'],
                    'Points': current_df['points'],
                    'Wins': current_df['wins'],
                    'Podiums': current_df['podiums'],
                    'Win %': current_df['win_rate'],
                    'Podium %': current_df['podium_rate'],
                    'Top 10 %': current_df['top_10_rate'],
                    'DNF %': current_df['dnf_rate'],
                    'Exp': current_df['years_in_f3'],
                    'DoB': current_df['dob'],
                    'Age': current_df['age'],
                    'Avg_Pos_Diff': current_df['avg_pos_vs_teammates'],
                    'Teammate_Battles': current_df['teammate_battles'],
                    'Participation %': current_df['participation_rate'],
                    'team': current_df['team'],
                    'team_pos': current_df['team_pos'],
                    'team_points': current_df['team_points'],
                    'points_vs_team_strength': current_df['points_vs_team_strength'],
                    'pos_vs_team_strength': current_df['pos_vs_team_strength'],
                    'Raw_Prob': raw_probas,
                    'Empirical_%': empirical_pct,
                    'Pred': (raw_probas > 0.5).astype(int)
                }).sort_values('Empirical_%', ascending=False)

                print(f"\n{name} Predictions:")
                print("-" * 50)
                print(results.head(10).to_string(index=False, float_format='%.3f'))

            except Exception as e:
                print(f"Error with {name} model: {e}")
                continue

    if results is not None:
        return results
    return pd.DataFrame()


print("Loading F3 data...")
f3_df = load_standings_data('F3', 'drivers')

print("Loading F2 data...")
f2_df = load_standings_data('F2', 'drivers')

print("Loading F3 team championship data...")
f3_team_df = load_standings_data('F3', 'teams')

print("Loading F3 qualifying data...")
f3_qualifying_df = load_qualifying_data('F3')

if f3_df.empty or f2_df.empty:
    print("No F2/F3 data found. Check file paths.")
    exit()

print("Enhancing F3 data with team championship metrics...")
f3_df = enhance_with_team_data(f3_df, f3_team_df)

print("Adding qualifying features...")
f3_df = calculate_qualifying_features(f3_df, f3_qualifying_df)

print("Creating target variable based on F2 participation...")
f3_df = create_target_variable(f3_df, f2_df)

print("Engineering features...")
features_df = engineer_features(f3_df)
features_df['moved_to_f2'] = f3_df['moved_to_f2']

print("Training all models...")
models, deep_models, X_test, y_test, feature_cols, scaler = train_models(
    features_df)

print("Making predictions for F3 2025 drivers...")
all_models = {
    'Traditional': models,
    'Deep Learning': deep_models
}
predict_drivers(all_models, features_df, feature_cols, scaler)
