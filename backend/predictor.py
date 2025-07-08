import json
import numpy as np
import os
import pandas as pd
import random
import re
import torch.optim as optim
import torch.nn as nn
import torch

from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from loader import load_all_data, load_qualifying_data

SEED = 69
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

NOT_PARTICIPATED_CODES = ['nan', 'DNS', 'WD', 'DNQ', 'DNA', 'C', 'EX']
RETIREMENT_CODES = ['Ret', 'NC', 'DSQ', 'DSQP']
CURRENT_YEAR = datetime.now().year


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


def extract_position(result_str):
    """Extract numeric position from result string."""
    if not result_str or result_str in NOT_PARTICIPATED_CODES:
        return None

    try:
        clean_str = result_str.split()[0].replace('†', '').replace('F', '').replace('P', '')
        return int(float(clean_str))
    except (ValueError, IndexError):
        return None


def calculate_participation_stats(df, race_cols):
    """Calculate participation statistics for a dataframe."""
    stats = []

    for _, row in df.iterrows():
        participated_races = 0
        positions = []

        for col in race_cols:
            result = str(row[col]).strip()
            if not result or result in NOT_PARTICIPATED_CODES:
                continue

            participated_races += 1

            if any(x in result for x in RETIREMENT_CODES):
                continue
            else:
                pos = extract_position(result)
                if pos:
                    positions.append(pos)

        stats.append({
            'Driver': row['Driver'],
            'year': row['year'],
            'participated_races': participated_races,
            'positions': positions,
        })

    return stats


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

            # Get teammates (excluding current driver)
            teammates_df = team_df[team_df['Driver'] != driver]

            if teammates_df.empty:
                continue

            # Calculate head-to-head record
            h2h_wins = 0
            h2h_total = 0

            for race_col in race_cols:
                driver_pos = extract_position(str(driver_row[race_col]).strip())
                if not driver_pos:
                    continue

                # Compare with each teammate
                for _, teammate_row in teammates_df.iterrows():
                    teammate_pos = extract_position(str(teammate_row[race_col]).strip())

                    if teammate_pos:
                        h2h_total += 1
                        if driver_pos < teammate_pos:
                            h2h_wins += 1

            h2h_win_rate = h2h_wins / h2h_total if h2h_total > 0 else 0.5
            is_multi_team = driver_row.get('team_count', 1) > 1

            team_performance.append({
                'Driver': driver,
                'year': year,
                'Team': team,
                'teammate_h2h_rate': h2h_win_rate,
                'is_multi_team': is_multi_team
            })

    # Convert to DataFrame and merge with original
    if team_performance:
        team_perf_df = pd.DataFrame(team_performance)
        df = df.merge(
            team_perf_df[['Driver', 'year', 'teammate_h2h_rate', 'is_multi_team']],
            on=['Driver', 'year'], how='left'
        )

    # Fill defaults
    defaults = {'teammate_h2h_rate': 0, 'is_multi_team': False}
    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].infer_objects(copy=False)
        else:
            df[col] = default

    return df


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


def add_driver_features(features_df, f3_df):
    """Add features from cached JSON profiles."""
    profiles_dir = "data/driver_profiles"

    # Load cached profiles
    profiles = {}
    if os.path.exists(profiles_dir):
        for driver in f3_df['Driver'].unique():
            profile_file = os.path.join(
                profiles_dir, get_driver_filename(driver))
            try:
                if os.path.exists(profile_file):
                    with open(profile_file, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)

                        # Check if driver was successfully scraped
                        if profile_data.get('scraped', True):
                            profiles[driver] = profile_data
                        else:
                            # Driver exists but wasn't scraped - treat as no data
                            profiles[driver] = {'dob': None, 'nationality': None}
                else:
                    profiles[driver] = {'dob': None, 'nationality': None}
            except BaseException:
                profiles[driver] = {'dob': None, 'nationality': None}

    # Add features
    feature_data = []
    for _, row in features_df.iterrows():
        profile = profiles.get(row['driver'], {})
        feature_data.append({
            'dob': profile.get('dob'),
            'age': calculate_age(profile.get('dob'), row['year']),
            'nationality': profile.get('nationality', 'Unknown')
        })

    for key in ['dob', 'age', 'nationality']:
        features_df[key] = [item[key] for item in feature_data]

    features_df['age'] = features_df['age'].fillna(features_df['age'].median())
    return features_df


def calculate_qualifying_features(df, qualifying_df):
    """Calculate qualifying-based features for drivers."""
    if qualifying_df.empty:
        df['avg_quali_pos'] = np.nan
        df['std_quali_pos'] = np.nan
        return df

    # Calculate qualifying statistics for each driver-year combination
    qualifying_stats = []

    for (driver, year), driver_data in qualifying_df.groupby(['Driver', 'year']):
        positions = []
        # Define priority order for qualifying position columns
        position_columns = [
            'Grid',     # Highest priority
            'GridFR',  # Secondary priority
            'R2',       # Tertiary priority
            'Pos',      # Fallback options
            'Pos.'
        ]

        # Extract qualifying positions from all rounds
        for _, row in driver_data.iterrows():
            # Check columns in priority order
            for col in position_columns:
                if col in row and pd.notna(row[col]):
                    # Handle numeric values directly
                    if isinstance(row[col], (int, float)):
                        positions.append(int(row[col]))
                        break

                    # Convert to string and clean
                    str_value = str(row[col]).strip()

                    # Skip non-participation codes
                    # Extract numeric position using regex (max 2 digits)
                    if str_value not in NOT_PARTICIPATED_CODES:
                        match = re.search(r'\b\d{1,2}\b', str_value)
                        if match:
                            positions.append(int(match.group()))
                            break

        qualifying_stats.append({
            'Driver': driver,
            'year': year,
            'avg_quali_pos': np.mean(positions) if positions else np.nan,
            'std_quali_pos': np.std(positions) if len(positions) > 1 else 0,
            'pole_rate': sum(1 for p in positions if p == 1) / len(positions) if positions else np.nan,  # noqa: 501
            'top_10_starts_rate': sum(1 for p in positions if p <= 10) / len(positions) if positions else np.nan,  # noqa: 501
        })

    # Convert to DataFrame and merge with main data
    quali_stats_df = pd.DataFrame(qualifying_stats)
    if not quali_stats_df.empty:
        df = df.merge(quali_stats_df, on=['Driver', 'year'], how='left')

    # Fill missing values
    quali_cols = ['avg_quali_pos', 'std_quali_pos', 'pole_rate', 'top_10_starts_rate']
    for col in quali_cols:
        if col not in df.columns:
            df[col] = 0 if 'starts' in col or 'sessions' in col else np.nan

    return df


def engineer_features(df):
    """Create features for ML models"""
    if df.empty:
        return pd.DataFrame()

    df = calculate_teammate_performance(df)
    df = df.sort_values(by=['Driver', 'year'])
    df['experience'] = df.groupby('Driver').cumcount()

    features_df = pd.DataFrame({
        'year': df['year'],
        'driver': df['Driver'],
        'final_pos': pd.to_numeric(df['Pos'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'),
        'points': pd.to_numeric(df['Points'], errors='coerce').fillna(0),
        'experience': df['experience'],
        'series_type': df.get('series_type', 'Unknown'),
        'team': df.get('Team'),
        'team_pos': df.get('team_pos', np.nan),
        'team_pos_per': df.get('team_pos_per', 0.5),
        'team_points': df.get('team_points', 0),
        'teammate_h2h_rate': df.get('teammate_h2h_rate', 0.5),
        'avg_quali_pos': df.get('avg_quali_pos', np.nan),
        'std_quali_pos': df.get('std_quali_pos', np.nan),
        'pole_rate': df.get('pole_rate', np.nan),
        'top_10_starts_rate': df.get('top_10_starts_rate', np.nan),
    })

    features_df['is_f3_european'] = (features_df['series_type'] == 'F3_European').astype(int)

    # Calculate race statistics for each driver
    race_stats = []
    cache_key_to_data = {}

    for _, row in df.iterrows():
        # Create cache key for this year/series combination
        cache_key = (row['year'], row.get('series_type', 'F3_Main'))

        # Cache race data
        if cache_key not in cache_key_to_data:
            year_series_data = df[
                (df['year'] == row['year']) &
                (df.get('series_type', 'F3_Main') == row.get('series_type', 'F3_Main'))
            ]
            race_cols = get_race_columns(year_series_data)
            cache_key_to_data[cache_key] = (race_cols)

        race_cols = cache_key_to_data[cache_key]

        stats = {
            'wins': 0, 'podiums': 0, 'top_10s': 0, 'dnfs': 0,
            'races_completed': 0, 'finish_positions': [],
        }

        for col in race_cols:
            if col not in row or pd.isna(row[col]):
                continue

            result = str(row[col]).strip()
            if not result or result in NOT_PARTICIPATED_CODES:
                continue

            stats['races_completed'] += 1

            if any(x in result for x in RETIREMENT_CODES):
                if result != 'NC':
                    stats['dnfs'] += 1
                continue

            try:
                pos = extract_position(result)
                stats['finish_positions'].append(pos)

                if pos == 1:
                    stats['wins'] += 1
                    stats['podiums'] += 1
                    stats['top_10s'] += 1
                elif pos <= 3:
                    stats['podiums'] += 1
                    stats['top_10s'] += 1
                elif pos <= 10:
                    stats['top_10s'] += 1
            except Exception as e:
                print(f"Error processing position '{result}': {e}")
                continue

        stats['participation_rate'] = stats['races_completed'] / len(race_cols) if race_cols else 0
        stats['avg_finish_pos'] = np.mean(stats['finish_positions']) if stats['finish_positions'] else np.nan # noqa: 501
        stats['std_finish_pos'] = np.std(stats['finish_positions']) if stats['finish_positions'] else np.nan # noqa: 501

        race_stats.append(stats)

    # Add race statistics to features
    for stat_name in ['wins', 'podiums', 'top_10s', 'dnfs', 'races_completed',
                      'participation_rate', 'avg_finish_pos', 'std_finish_pos']:
        features_df[stat_name] = [stats[stat_name] for stats in race_stats]

    # Filter out drivers with no races
    features_df = features_df[features_df['races_completed'] > 0]

    features_df = add_driver_features(features_df, f3_df)

    features_df['win_rate'] = features_df['wins'] / features_df['races_completed']
    features_df['podium_rate'] = features_df['podiums'] / features_df['races_completed']
    features_df['top_10_rate'] = features_df['top_10s'] / features_df['races_completed']
    features_df['dnf_rate'] = features_df['dnfs'] / features_df['races_completed']
    features_df['points_share'] = features_df['points'] / (features_df['team_points'] + 1)

    return features_df


def create_target_variable(f3_df, f2_df):
    """Create target variable for F2 participation after last F3 season."""
    if f3_df.empty or f2_df.empty:
        f3_df['promoted'] = np.nan
        return f3_df

    # Initialize target column
    f3_df['promoted'] = 0
    max_f2_year = f2_df['year'].max()

    # Get last F3 season for each driver
    last_f3_seasons = f3_df.groupby('Driver')['year'].max().reset_index()

    # Process F2 data to determine participation
    f2_participation = []
    for year, year_df in f2_df.groupby('year'):
        race_cols = get_race_columns(year_df)
        if not race_cols:
            continue

        participation_stats = calculate_participation_stats(year_df, race_cols)
        threshold = 0 if year == CURRENT_YEAR else len(race_cols) * 0.4

        for stat in participation_stats:
            f2_participation.append({
                'driver': stat['Driver'],
                'year': year,
                'participated': stat['participated_races'] > threshold
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
            f3_df.at[idx, 'promoted'] = moved_drivers[(driver, year)]

    return f3_df


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


def train_models(df):
    """Training function with temporal split and stratification."""
    if df.empty:
        print("No data available for training")
        return {}, {}, None, None, None, None, None, None

    df_clean = df.dropna(subset=['promoted', 'final_pos', 'points'])
    feature_cols = [
        'avg_finish_pos', 'std_finish_pos',
        'avg_quali_pos', 'std_quali_pos',
        'win_rate', 'podium_rate', 'top_10_rate',
        'participation_rate', 'dnf_rate',
        'experience', 'age',
        'teammate_h2h_rate', 'points_share',
        'pole_rate', 'top_10_starts_rate',
    ]

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['promoted']
    years = df_clean['year']

    # print(f"Dataset size: {len(X)} drivers")
    # print(f"F2 progressions: {y.sum()} ({y.mean():.2%})")
    # print(f"Year range: {years.min()} - {years.max()}")

    # Use 80% of years for training, 20% for testing
    unique_years = sorted(years.unique())
    n_train_years = int(len(unique_years) * 0.8)
    train_years = unique_years[:n_train_years]
    test_years = unique_years[n_train_years:]

    # Split data based on temporal cutoff
    train_mask = years.isin(train_years)
    test_mask = years.isin(test_years)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    # Check class distribution in both sets
    # print(f"Training: {len(X_train)} samples, {y_train.sum()} promotions ({y_train.mean():.2%})")
    # print(f"Test: {len(X_test)} samples, {y_test.sum()} promotions ({y_test.mean():.2%})")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Traditional ML pipelines
    traditional_pipelines = {
        'Random Forest': ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('classifier', RandomForestClassifier(random_state=SEED))
        ]),
        'Logistic Regression': ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('classifier', LogisticRegression(random_state=SEED, max_iter=1000))
        ]),
        'LightGBM': ImbPipeline([
            ('classifier', LGBMClassifier(random_state=SEED, class_weight='balanced', verbosity=-1))
        ]),
        'MLP': ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('classifier', MLPClassifier(random_state=SEED, max_iter=1000, early_stopping=True))
        ]),
        'SVM': ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('classifier', SVC(random_state=SEED, probability=True))
        ]),
    }

    results = {}

    print("\n" + "=" * 50)
    print("TRAINING MODELS")
    print("=" * 50)

    # Train traditional models
    for name, pipeline in traditional_pipelines.items():
        print(f"\nTraining {name}:")
        print("-" * 40)

        # Fit and evaluate on test set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        probas_test = pipeline.predict_proba(X_test)[:, 1]

        # Calibration
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(probas_test, y_test)
        pipeline.calibrator = iso_reg

        print("\nTest Set Results:")
        print(classification_report(y_test, y_pred))

        pr_auc = average_precision_score(y_test, probas_test)
        print(f"Precision-Recall AUC: {pr_auc:.4f}")

        results[name] = pipeline

    # Train PyTorch Model models
    print("\nTraining PyTorch Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to PyTorch tensors
    X_train_torch = torch.FloatTensor(X_train_scaled).to(device)
    y_train_torch = torch.FloatTensor(y_train.values).to(device)
    X_test_torch = torch.FloatTensor(X_test_scaled).to(device)

    pytorch_model = RacingPredictor(X_train_scaled.shape[1]).to(device)

    # Calculate class weights for PyTorch
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Validation split
    n_train_samples = len(X_train_torch)
    val_split_idx = int(n_train_samples * 0.8)

    # Create sequential indices for tensor slicing
    train_indices = list(range(val_split_idx))
    val_indices = list(range(val_split_idx, n_train_samples))

    X_train_sub = X_train_torch[train_indices]
    y_train_sub = y_train_torch[train_indices]
    X_val = X_train_torch[val_indices]
    y_val = y_train_torch[val_indices]

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):
        pytorch_model.train()
        optimizer.zero_grad()

        outputs = pytorch_model(X_train_sub).squeeze()
        loss = criterion(outputs, y_train_sub)
        loss.backward()
        optimizer.step()

        # Validation
        pytorch_model.eval()
        with torch.no_grad():
            val_outputs = pytorch_model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = pytorch_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    # Load best model and evaluate
    pytorch_model.load_state_dict(best_state_dict)
    pytorch_model.eval()

    with torch.no_grad():
        logits = pytorch_model(X_test_torch)
        raw_probas_torch = torch.sigmoid(logits).cpu().numpy().flatten()

    # Calibrate PyTorch model
    iso_reg_torch = IsotonicRegression(out_of_bounds='clip')
    iso_reg_torch.fit(raw_probas_torch, y_test)
    pytorch_model.calibrator = iso_reg_torch

    pytorch_pred = (raw_probas_torch > 0.5).astype(int)
    print("PyTorch Classification Report:")
    print(classification_report(y_test, pytorch_pred))
    pr_auc_torch = average_precision_score(y_test, raw_probas_torch)
    print(f"PyTorch Precision-Recall AUC: {pr_auc_torch:.4f}")
    results['PyTorch'] = pytorch_model

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return results, X_test, y_test, feature_cols, scaler, X_train, y_train


def predict_drivers(models, df, feature_cols, scaler=None):
    """Make predictions for F3 2025 drivers"""
    current_year = CURRENT_YEAR
    current_df = df[df['year'] == current_year].copy()
    if current_df.empty:
        current_year = df['year'].max()
        current_df = df[df['year'] == current_year].copy()
    if current_df.empty:
        print("No current data found for predictions")
        return pd.DataFrame()

    X_current = current_df[feature_cols].fillna(0)
    results = None

    for name, model in models.items():
        print(f"\n{name} Predictions:")
        print("=" * 70)

        try:
            # Get raw probabilities based on model type
            if name == 'PyTorch':
                if scaler is not None:
                    X_processed = scaler.transform(X_current)
                else:
                    X_processed = X_current
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_torch = torch.FloatTensor(X_processed).to(device)
                    logits = model(X_torch)
                    raw_probas = torch.sigmoid(logits).cpu().numpy().flatten()
            else:  # Traditional models
                X_processed = X_current
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
                'Participation %': current_df['participation_rate'],
                'Exp': current_df['experience'],
                'DoB': current_df['dob'],
                'Age': current_df['age'],
                'teammate_h2h_rate': current_df['teammate_h2h_rate'],
                'Pole %': current_df['pole_rate'],
                'top_10_starts_rate': current_df['top_10_starts_rate'],
                'team': current_df['team'],
                'team_pos': current_df['team_pos'],
                'team_points': current_df['team_points'],
                'points_share': current_df['points_share'],
                'Raw_Prob': raw_probas,
                'Empirical_%': empirical_pct
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


print("Loading F3 qualifying data...")
f3_qualifying_df = load_qualifying_data('F3')

f2_df, f3_df = load_all_data()

print("Adding qualifying features...")
f3_df = calculate_qualifying_features(f3_df, f3_qualifying_df)

print("Creating target variable based on F2 participation...")
f3_df = create_target_variable(f3_df, f2_df)

print("Engineering features...")
features_df = engineer_features(f3_df)
features_df['promoted'] = f3_df['promoted']

print("Training all models...")
models, X_test, y_test, feature_cols, scaler, X_train, y_train = train_models(
    features_df)

print("Making predictions for F3 2025 drivers...")
predict_drivers(models, features_df, feature_cols, scaler)
