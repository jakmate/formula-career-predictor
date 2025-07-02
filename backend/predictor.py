import json
import numpy as np
import os
import pandas as pd
import random
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from loader import load_all_data, load_qualifying_data

SEED = 69
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

NOT_PARTICIPATED_CODES = ['', 'DNS', 'WD', 'DNQ', 'DNA', 'C', 'EX']
RETIREMENT_CODES = ['Ret', 'NC', 'DSQ', 'DSQP']
CURRENT_YEAR = datetime.now().year

COLUMN_MAPPING = {
    'Driver name': 'Driver',
    'Drivers': 'Driver',
    'Entrant': 'Team'
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
            pos_differences = []

            for race_col in race_cols:
                driver_pos = extract_position(str(driver_row[race_col]).strip())
                if not driver_pos:
                    continue

                # Compare with each teammate
                teammate_positions = []
                for _, teammate_row in teammates_df.iterrows():
                    teammate_pos = extract_position(str(teammate_row[race_col]).strip())

                    if teammate_pos:
                        teammate_positions.append(teammate_pos)
                        h2h_total += 1
                        if driver_pos < teammate_pos:
                            h2h_wins += 1

                if teammate_positions:
                    avg_teammate_pos = np.mean(teammate_positions)
                    pos_differences.append(avg_teammate_pos - driver_pos)

            avg_pos_difference = np.mean(pos_differences) if pos_differences else 0
            is_multi_team = driver_row.get('team_count', 1) > 1

            if is_multi_team and h2h_total > 0:
                avg_pos_difference *= 0.7
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
    if team_performance:
        team_perf_df = pd.DataFrame(team_performance)
        df = df.merge(
            team_perf_df[['Driver', 'year', 'avg_pos_vs_teammates',
                          'teammate_battles', 'is_multi_team']],
            on=['Driver', 'year'], how='left'
        )

    # Fill defaults
    defaults = {'avg_pos_vs_teammates': 0, 'teammate_battles': 0, 'is_multi_team': False}
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

    qualifying_df = apply_column_mapping(qualifying_df)
    qualifying_df = clean_string_columns(qualifying_df, ['Driver'])

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
            'std_quali_pos': np.std(positions) if len(positions) > 1 else 0
        })

    # Convert to DataFrame and merge with main data
    quali_stats_df = pd.DataFrame(qualifying_stats)

    if not quali_stats_df.empty:
        df = df.merge(quali_stats_df, on=['Driver', 'year'], how='left')
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
        'avg_pos_vs_teammates': df.get('avg_pos_vs_teammates', 0),
        'teammate_battles': df.get('teammate_battles', 0),
        'avg_quali_pos': df.get('avg_quali_pos', np.nan),
        'std_quali_pos': df.get('std_quali_pos', np.nan)
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
            field_size = len(year_series_data)
            cache_key_to_data[cache_key] = (race_cols, field_size)

        race_cols, field_size = cache_key_to_data[cache_key]

        stats = {
            'wins': 0, 'podiums': 0, 'points_finishes': 0, 'dnfs': 0,
            'races_completed': 0, 'field_size': field_size,
            'finish_positions': []
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
                    stats['points_finishes'] += 1
                elif pos <= 3:
                    stats['podiums'] += 1
                    stats['points_finishes'] += 1
                elif pos <= 10:
                    stats['points_finishes'] += 1
            except Exception as e:
                print(f"Error processing position '{result}': {e}")
                continue

        stats['participation_rate'] = stats['races_completed'] / len(race_cols) if race_cols else 0
        stats['avg_finish_pos'] = np.mean(stats['finish_positions']) if stats['finish_positions'] else np.nan # noqa: 501
        stats['std_finish_pos'] = np.std(stats['finish_positions']) if stats['finish_positions'] else np.nan # noqa: 501

        race_stats.append(stats)

    # Add race statistics to features
    for stat_name in ['wins', 'podiums', 'points_finishes', 'dnfs', 'races_completed',
                      'participation_rate', 'field_size', 'avg_finish_pos', 'std_finish_pos']:
        features_df[stat_name] = [stats[stat_name] for stats in race_stats]

    # Filter out drivers with no races
    features_df = features_df[features_df['races_completed'] > 0]

    features_df = add_driver_features(features_df, f3_df)

    features_df['win_rate'] = features_df['wins'] / features_df['races_completed']
    features_df['podium_rate'] = features_df['podiums'] / features_df['races_completed']
    features_df['dnf_rate'] = features_df['dnfs'] / features_df['races_completed']
    features_df['top_10_rate'] = features_df['points_finishes'] / features_df['races_completed']
    features_df['points_vs_team_strength'] = features_df['points'] / (features_df['team_points'] + 1) # noqa: 501
    features_df['pos_vs_team_strength'] = features_df['final_pos'] * features_df['team_pos_per']

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
        threshold = 0 if year == CURRENT_YEAR else len(race_cols) * 0.5

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


def create_tensorflow_dnn(input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3):
    # Input layer
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden_layers[0], activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate)
    ])

    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['precision', 'recall']
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


def train_models(df):
    """Training function for all model types."""
    if df.empty:
        print("No data available for training")
        return {}, {}, None, None, None

    df_clean = df.dropna(subset=['promoted', 'final_pos', 'points'])
    feature_cols = [
        'avg_finish_pos', 'std_finish_pos',
        'avg_quali_pos', 'std_quali_pos',
        'win_rate', 'podium_rate', 'top_10_rate',
        'participation_rate', 'dnf_rate',
        'experience', 'age',
        'avg_pos_vs_teammates', 'teammate_battles',
        'pos_vs_team_strength', 'points_vs_team_strength'
    ]

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['promoted']

    print(f"Dataset size: {len(X)} drivers")
    print(f"F2 progressions: {y.sum()} ({y.mean():.2%})")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train_scaled, X_test_scaled, _, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Traditional ML pipelines
    traditional_pipelines = {
        'Random Forest': ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('classifier', RandomForestClassifier(
                random_state=SEED, class_weight='balanced'))
        ]),
        'Logistic Regression': ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=SEED, class_weight='balanced', max_iter=10000))
        ]),
        'XGBoost': ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('classifier', xgb.XGBClassifier(
                random_state=SEED, eval_metric='logloss',
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            ))
        ]),
        'MLP': ImbPipeline([
            ('smote', SMOTE(random_state=SEED)),
            ('classifier', MLPClassifier(random_state=SEED, max_iter=10000))
        ])
    }

    traditional_results = {}
    deep_results = {}

    # Train traditional models
    print("\n" + "=" * 50)
    print("TRAINING TRADITIONAL MODELS")
    print("=" * 50)

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

        traditional_results[name] = pipeline

    # Train deep learning models
    print("\n" + "=" * 50)
    print("TRAINING DEEP LEARNING MODELS")
    print("=" * 50)

    # Calculate class weights for neural networks
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Keras DNN
    print("\nTraining Keras Deep Neural Network...")
    dnn_model = create_tensorflow_dnn(X_train_scaled.shape[1])

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5)
    ]

    dnn_model.fit(
        X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32,
        class_weight=class_weight_dict, callbacks=callbacks, verbose=0
    )

    # Evaluate and calibrate
    raw_probas_dnn = dnn_model.predict(X_test_scaled, verbose=0).flatten()
    iso_reg_dnn = IsotonicRegression(out_of_bounds='clip')
    iso_reg_dnn.fit(raw_probas_dnn, y_test)
    dnn_model.calibrator = iso_reg_dnn

    dnn_pred = (raw_probas_dnn > 0.5).astype(int)
    print("Keras DNN Classification Report:")
    print(classification_report(y_test, dnn_pred))
    pr_auc_dnn = average_precision_score(y_test, raw_probas_dnn)
    print(f"Keras DNN Precision-Recall AUC: {pr_auc_dnn:.4f}")
    deep_results['Keras_DNN'] = dnn_model

    # PyTorch Model
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
    val_size = int(0.2 * len(X_train_torch))
    indices = torch.randperm(len(X_train_torch))
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    X_train_sub = X_train_torch[train_idx]
    y_train_sub = y_train_torch[train_idx]
    X_val = X_train_torch[val_idx]
    y_val = y_train_torch[val_idx]

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
    deep_results['PyTorch'] = pytorch_model

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Traditional Models: {list(traditional_results.keys())}")
    print(f"Deep Learning Models: {list(deep_results.keys())}")

    return traditional_results, deep_results, X_test, y_test, feature_cols, scaler, X_train, y_train


def predict_drivers(all_models, df, feature_cols, scaler=None):
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
                    'Exp': current_df['experience'],
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


def analyze_feature_importance(models, deep_models, X_test, y_test, feature_cols, scaler=None):
    """Analyze feature importance across all models"""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    importance_results = {}

    # Traditional models
    for name, model in models.items():
        print(f"\n{name} Feature Analysis:")
        print("-" * 50)

        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            # Tree-based models (RF, XGBoost)
            importances = model.named_steps['classifier'].feature_importances_
        elif hasattr(model.named_steps['classifier'], 'coef_'):
            # Linear models (LogReg)
            importances = np.abs(model.named_steps['classifier'].coef_[0])
        else:
            # MLP
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=SEED
            )
            importances = perm_importance.importances_mean

        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print(importance_df.head(10))
        importance_results[name] = importance_df

    # Deep learning models
    if deep_models and scaler is not None:
        X_test_scaled = scaler.transform(X_test)

        for name, model in deep_models.items():
            print(f"\n{name} Feature Analysis:")
            print("-" * 50)

            try:
                # Create wrapper function for deep learning models
                def dl_predict_wrapper(X):
                    if 'Keras' in name:
                        return model.predict(X, verbose=0).flatten()
                    elif 'PyTorch' in name:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model.eval()
                        with torch.no_grad():
                            X_torch = torch.FloatTensor(X).to(device)
                            logits = model(X_torch)
                            return torch.sigmoid(logits).cpu().numpy().flatten()

                importances = []
                baseline_score = average_precision_score(y_test, dl_predict_wrapper(X_test_scaled))

                for i, feature in enumerate(feature_cols):
                    X_permuted = X_test_scaled.copy()
                    np.random.shuffle(X_permuted[:, i])
                    permuted_score = average_precision_score(y_test, dl_predict_wrapper(X_permuted))
                    importance = baseline_score - permuted_score
                    importances.append(importance)

                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                print(importance_df.head(10))
                importance_results[name] = importance_df

            except Exception as e:
                print(f"Error analyzing {name}: {e}")

    return importance_results


def analyze_negative_features(models, deep_models, X_train, y_train,
                              X_test, y_test, feature_cols, scaler=None):
    """Identify features that negatively impact performance"""
    print("\n" + "="*70)
    print("NEGATIVE FEATURE IMPACT ANALYSIS")
    print("="*70)

    negative_impact = {}

    # Traditional models
    for name, model in models.items():
        print(f"\n{name} - Features with Negative Impact:")
        print("-" * 50)

        # Get baseline performance
        baseline_score = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        feature_impacts = []

        # Test removing each feature
        for feature in feature_cols:
            remaining_features = [f for f in feature_cols if f != feature]
            X_train_reduced = X_train[remaining_features]
            X_test_reduced = X_test[remaining_features]

            try:
                # Clone and retrain model without this feature
                temp_model = clone(model)
                temp_model.fit(X_train_reduced, y_train)
                score_without = average_precision_score(
                    y_test,
                    temp_model.predict_proba(X_test_reduced)[:, 1]
                )
                impact = score_without - baseline_score
                feature_impacts.append({
                    'Feature': feature,
                    'Impact': impact,
                    'Baseline_PR_AUC': baseline_score,
                    'Without_Feature_PR_AUC': score_without
                })
            except Exception as e:
                print(f"Error testing {feature}: {e}")
                continue

        impact_df = pd.DataFrame(feature_impacts).sort_values('Impact', ascending=False)
        negative_features = impact_df[impact_df['Impact'] > 0]
        if not negative_features.empty:
            print("Features that hurt performance (removing them improves PR-AUC):")
            print(negative_features.head())
        else:
            print("No clearly negative features found")

        negative_impact[name] = impact_df

    # Deep learning models
    if deep_models and scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model in deep_models.items():
            print(f"\n{name} - Features with Negative Impact:")
            print("-" * 50)

            try:
                # Get baseline score
                if 'Keras' in name:
                    baseline_probas = model.predict(X_test_scaled, verbose=0).flatten()
                elif 'PyTorch' in name:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.eval()
                    with torch.no_grad():
                        X_torch = torch.FloatTensor(X_test_scaled).to(device)
                        logits = model(X_torch)
                        baseline_probas = torch.sigmoid(logits).cpu().numpy().flatten()

                baseline_score = average_precision_score(y_test, baseline_probas)
                feature_impacts = []

                for i, feature in enumerate(feature_cols):
                    # Create reduced datasets by removing feature column
                    feature_indices = [j for j in range(len(feature_cols)) if j != i]
                    X_train_reduced = X_train_scaled[:, feature_indices]
                    X_test_reduced = X_test_scaled[:, feature_indices]

                    try:
                        # Retrain model without this feature
                        if 'Keras' in name:
                            temp_model = create_tensorflow_dnn(X_train_reduced.shape[1])
                            temp_model.fit(
                                X_train_reduced, y_train, epochs=50,
                                batch_size=32, verbose=0
                            )
                            probas_without = temp_model.predict(X_test_reduced, verbose=0).flatten()
                        elif 'PyTorch' in name:
                            temp_model = RacingPredictor(X_train_reduced.shape[1]).to(device)
                            # Quick training - simplified
                            criterion = nn.BCEWithLogitsLoss()
                            optimizer = optim.Adam(temp_model.parameters(), lr=0.001)

                            X_train_torch = torch.FloatTensor(X_train_reduced).to(device)
                            y_train_torch = torch.FloatTensor(y_train.values).to(device)

                            for epoch in range(50):
                                temp_model.train()
                                optimizer.zero_grad()
                                outputs = temp_model(X_train_torch).squeeze()
                                loss = criterion(outputs, y_train_torch)
                                loss.backward()
                                optimizer.step()

                            temp_model.eval()
                            with torch.no_grad():
                                X_test_torch = torch.FloatTensor(X_test_reduced).to(device)
                                logits = temp_model(X_test_torch)
                                probas_without = torch.sigmoid(logits).cpu().numpy().flatten()

                        score_without = average_precision_score(y_test, probas_without)
                        impact = score_without - baseline_score

                        feature_impacts.append({
                            'Feature': feature,
                            'Impact': impact,
                            'Baseline_PR_AUC': baseline_score,
                            'Without_Feature_PR_AUC': score_without
                        })

                    except Exception as e:
                        print(f"Error testing {feature}: {e}")
                        continue

                impact_df = pd.DataFrame(feature_impacts).sort_values('Impact', ascending=False)
                negative_features = impact_df[impact_df['Impact'] > 0]
                if not negative_features.empty:
                    print("Features that hurt performance (removing them improves PR-AUC):")
                    print(negative_features.head())
                else:
                    print("No clearly negative features found")

                negative_impact[name] = impact_df

            except Exception as e:
                print(f"Error analyzing {name}: {e}")

    return negative_impact


def recursive_feature_elimination(models, X_train, y_train, X_test, y_test, feature_cols):
    """Use RFE to find optimal feature subset"""
    print("\n" + "="*70)
    print("RECURSIVE FEATURE ELIMINATION")
    print("="*70)

    rfe_results = {}

    for name, model in models.items():
        print(f"\n{name} - Recursive Feature Elimination:")
        print("-" * 50)

        # Try different numbers of features
        best_score = 0
        best_n_features = len(feature_cols)
        scores_by_n_features = {}

        for n_features in range(3, len(feature_cols) + 1, 2):
            try:
                if 'MLP' in name:
                    # Get permutation importance
                    perm_importance = permutation_importance(
                        model, X_train, y_train, n_repeats=10, random_state=SEED
                    )

                    # Select top n_features based on permutation importance
                    feature_importance = perm_importance.importances_mean
                    top_indices = np.argsort(feature_importance)[-n_features:]
                    selected_features = [feature_cols[i] for i in top_indices]
                else:
                    # Create RFE selector
                    rfe = RFE(model.named_steps['classifier'], n_features_to_select=n_features)
                    rfe.fit(X_train, y_train)

                    # Get selected features
                    selected_features = [feature_cols[i] for i, selected in enumerate(rfe.support_) if selected]  # noqa: 501

                # Test performance
                X_test_selected = X_test[selected_features]
                X_train_selected = X_train[selected_features]

                # Clone and retrain model with selected features
                temp_model = clone(model)
                temp_model.fit(X_train_selected, y_train)
                score = average_precision_score(
                    y_test,
                    temp_model.predict_proba(X_test_selected)[:, 1]
                )

                scores_by_n_features[n_features] = {
                    'score': score,
                    'features': selected_features
                }

                if score > best_score:
                    best_score = score
                    best_n_features = n_features

                print(f"{n_features} features: PR-AUC = {score:.4f}")

            except Exception as e:
                print(f"Error with {n_features} features: {e}")
                continue

        print(f"Best: {best_n_features} features with PR-AUC = {best_score:.4f}")

        # Show which features were eliminated in the best configuration
        if best_n_features in scores_by_n_features:
            selected = scores_by_n_features[best_n_features]['features']
            eliminated = [f for f in feature_cols if f not in selected]
            print(f"Eliminated features: {eliminated}")

        rfe_results[name] = scores_by_n_features

    return rfe_results


def correlation_analysis(X, feature_cols):
    """Analyze feature correlations to identify redundant features"""
    print("\n" + "="*70)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*70)

    # Calculate correlation matrix
    corr_matrix = X[feature_cols].corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                high_corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })

    if high_corr_pairs:
        print("Highly correlated feature pairs (|correlation| > 0.8):")
        corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
        print(corr_df)

        print("\nConsider removing one feature from each highly correlated pair")
        return corr_df
    else:
        print("No highly correlated features found")
        return pd.DataFrame()


def feature_ablation_study(models, deep_models, X_train, y_train,
                           X_test, y_test, feature_cols, scaler=None):
    """Systematic feature ablation study"""
    print("\n" + "="*70)
    print("FEATURE ABLATION STUDY")
    print("="*70)

    ablation_results = {}

    # Define feature groups
    actual_features = set(feature_cols)
    feature_groups = {}

    position_features = [f for f in ['final_pos', 'avg_finish_pos', 'std_finish_pos'] if f in actual_features] # noqa: 501
    if position_features:
        feature_groups['position_features'] = position_features

    rate_features = [f for f in ['win_rate', 'podium_rate', 'dnf_rate', 'top_10_rate'] if f in actual_features] # noqa: 501
    if rate_features:
        feature_groups['rate_features'] = rate_features

    experience_features = [f for f in ['experience', 'age'] if f in actual_features]
    if experience_features:
        feature_groups['experience_features'] = experience_features

    teammate_features = [f for f in ['avg_pos_vs_teammates', 'teammate_battles'] if f in actual_features] # noqa: 501
    if teammate_features:
        feature_groups['teammate_features'] = teammate_features

    team_features = [f for f in ['team_pos', 'points_vs_team_strength', 'pos_vs_team_strength'] if f in actual_features] # noqa: 501
    if team_features:
        feature_groups['team_features'] = team_features

    quali_features = [f for f in ['avg_quali_pos', 'std_quali_pos'] if f in actual_features]
    if quali_features:
        feature_groups['qualifying_features'] = quali_features

    other_features = [f for f in ['participation_rate', 'is_f3_european'] if f in actual_features]
    if other_features:
        feature_groups['other_features'] = other_features

    print(f"Feature groups identified: {list(feature_groups.keys())}")

    # Traditional models
    for name, model in models.items():
        print(f"\n{name} - Feature Ablation:")
        print("-" * 50)

        # Get baseline performance with all features
        try:
            baseline_score = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        except Exception as e:
            print(f"Error getting baseline score for {name}: {e}")
            continue

        feature_ablations = []

        # Test performance when removing each feature group
        for group_name, group_features in feature_groups.items():
            remaining_features = [f for f in feature_cols if f not in group_features]

            try:
                X_train_reduced = X_train[remaining_features]
                X_test_reduced = X_test[remaining_features]

                temp_model = clone(model)
                temp_model.fit(X_train_reduced, y_train)
                score = average_precision_score(
                    y_test,
                    temp_model.predict_proba(X_test_reduced)[:, 1]
                )

                impact = baseline_score - score

                feature_ablations.append({
                    'Feature_Group': group_name,
                    'Features_Removed': group_features,
                    'Baseline_PR_AUC': baseline_score,
                    'Reduced_PR_AUC': score,
                    'Impact': impact
                })

                print(f"Removing {group_name}: PR AUC = {score:.4f} (impact: {impact:+.4f})")

            except Exception as e:
                print(f"Error removing {group_name}: {e}")
                continue

        if feature_ablations:
            ablation_results[name] = pd.DataFrame(feature_ablations).sort_values(
                'Impact',
                ascending=False
            )
        else:
            ablation_results[name] = pd.DataFrame()

    # Deep learning models
    if deep_models and scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model in deep_models.items():
            print(f"\n{name} - Feature Ablation:")
            print("-" * 50)

            try:
                # Get baseline score
                if 'Keras' in name:
                    baseline_probas = model.predict(X_test_scaled, verbose=0).flatten()
                elif 'PyTorch' in name:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.eval()
                    with torch.no_grad():
                        X_torch = torch.FloatTensor(X_test_scaled).to(device)
                        logits = model(X_torch)
                        baseline_probas = torch.sigmoid(logits).cpu().numpy().flatten()

                baseline_score = average_precision_score(y_test, baseline_probas)
                feature_ablations = []

                for group_name, group_features in feature_groups.items():
                    # Get indices of features to keep
                    remaining_feature_indices = [
                        i for i, f in enumerate(feature_cols) if f not in group_features
                    ]

                    X_train_reduced = X_train_scaled[:, remaining_feature_indices]
                    X_test_reduced = X_test_scaled[:, remaining_feature_indices]

                    try:
                        # Retrain model with reduced features
                        if 'Keras' in name:
                            temp_model = create_tensorflow_dnn(X_train_reduced.shape[1])
                            temp_model.fit(
                                X_train_reduced, y_train, epochs=50,
                                batch_size=32, verbose=0
                            )
                            probas_reduced = temp_model.predict(X_test_reduced, verbose=0).flatten()
                        elif 'PyTorch' in name:
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            temp_model = RacingPredictor(X_train_reduced.shape[1]).to(device)

                            criterion = nn.BCEWithLogitsLoss()
                            optimizer = optim.Adam(temp_model.parameters(), lr=0.001)

                            X_train_torch = torch.FloatTensor(X_train_reduced).to(device)
                            y_train_torch = torch.FloatTensor(y_train.values).to(device)

                            for epoch in range(50):
                                temp_model.train()
                                optimizer.zero_grad()
                                outputs = temp_model(X_train_torch).squeeze()
                                loss = criterion(outputs, y_train_torch)
                                loss.backward()
                                optimizer.step()

                            temp_model.eval()
                            with torch.no_grad():
                                X_test_torch = torch.FloatTensor(X_test_reduced).to(device)
                                logits = temp_model(X_test_torch)
                                probas_reduced = torch.sigmoid(logits).cpu().numpy().flatten()

                        score = average_precision_score(y_test, probas_reduced)
                        impact = baseline_score - score

                        feature_ablations.append({
                            'Feature_Group': group_name,
                            'Features_Removed': group_features,
                            'Baseline_PR_AUC': baseline_score,
                            'Reduced_PR_AUC': score,
                            'Impact': impact
                        })

                        print(f"Removing {group_name}: PR AUC = {score:.4f} (impact: {impact:+.4f})") # noqa: 501

                    except Exception as e:
                        print(f"Error removing {group_name}: {e}")
                        continue

                if feature_ablations:
                    ablation_results[name] = pd.DataFrame(feature_ablations).sort_values(
                        'Impact',
                        ascending=False
                    )
                else:
                    ablation_results[name] = pd.DataFrame()

            except Exception as e:
                print(f"Error with {name} ablation: {e}")

    return ablation_results


def comprehensive_feature_analysis(models, deep_models, X_train, y_train,
                                   X_test, y_test, feature_cols, scaler=None):
    """Run comprehensive feature analysis"""

    print(f"Starting comprehensive feature analysis with {len(feature_cols)} features:")
    print(f"Features: {feature_cols}")

    importance_results = analyze_feature_importance(models, deep_models, X_test,
                                                    y_test, feature_cols, scaler)
    negative_impact = analyze_negative_features(models, deep_models, X_train, y_train,
                                                X_test, y_test, feature_cols, scaler)
    correlation_results = correlation_analysis(pd.concat([X_train, X_test]), feature_cols)
    rfe_results = recursive_feature_elimination(models, X_train, y_train,
                                                X_test, y_test, feature_cols)
    ablation_results = feature_ablation_study(models, deep_models, X_train, y_train,
                                              X_test, y_test, feature_cols, scaler)

    print("\n" + "="*70)
    print("SUMMARY: POTENTIALLY PROBLEMATIC FEATURES")
    print("="*70)

    problematic_features = set()

    if not correlation_results.empty:
        for _, row in correlation_results.iterrows():
            problematic_features.add(row['Feature_2'])

    for model_name, ablation_df in ablation_results.items():
        if not ablation_df.empty:
            # Negative impact means removal improved performance
            negative_impact_groups = ablation_df[ablation_df['Impact'] < 0]
            for _, row in negative_impact_groups.iterrows():
                problematic_features.update(row['Features_Removed'])

    if problematic_features:
        print(f"Features that may be hindering performance: {list(problematic_features)}")
    else:
        print("No clearly problematic features identified")

    return {
        'importance': importance_results,
        'negative_impact': negative_impact,
        'correlation': correlation_results,
        'rfe': rfe_results,
        'ablation': ablation_results,
        'problematic_features': list(problematic_features)
    }


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
models, deep_models, X_test, y_test, feature_cols, scaler, X_train, y_train = train_models(
    features_df)

print("Making predictions for F3 2025 drivers...")
all_models = {
    'Traditional': models,
    'Deep Learning': deep_models
}
predict_drivers(all_models, features_df, feature_cols, scaler)

# comprehensive_feature_analysis(models, deep_models, X_train, y_train,
#                               X_test, y_test, feature_cols, scaler)
