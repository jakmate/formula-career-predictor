import glob
import json
import os
import numpy as np
import pandas as pd
import re
import torch.optim as optim
import torch.nn as nn
import torch
import xgboost as xgb
from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.models import Sequential
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def get_race_columns(df):
    """Identify race result columns based on track code patterns."""
    track_codes = set()
    for col in df.columns:
        parts = col.split()
        if parts:
            first_part = parts[0]
            if len(first_part) >= 3:
                code_candidate = first_part[:3]
                if code_candidate.isalpha() and code_candidate.isupper():
                    track_codes.add(code_candidate)

    race_columns = []
    for col in df.columns:
        parts = col.split()
        if parts:
            first_part = parts[0]
            if len(first_part) >= 3:
                code_candidate = first_part[:3]
                if code_candidate in track_codes:
                    race_columns.append(col)

    return race_columns


def load_and_combine_data(series='F3'):
    """Load and combine data for a racing series across years."""
    all_data = []
    series_dirs = glob.glob(f"{series}/*")

    for year_dir in series_dirs:
        year = os.path.basename(year_dir)
        try:
            year_int = int(year)
            driver_file = os.path.join(
                year_dir,
                f"{series.lower()}_{year}_drivers_standings.csv"
            )
            if os.path.exists(driver_file):
                df = pd.read_csv(driver_file)
                df['year'] = year_int
                df['series'] = series

                entries_file = os.path.join(
                    year_dir,
                    f"{series.lower()}_{year}_entries.csv"
                )
                if os.path.exists(entries_file):
                    entries_df = pd.read_csv(entries_file)

                    # Handle different column name variations
                    column_mapping = {
                        'Driver name': 'Driver',
                        'Drivers': 'Driver',
                        'Entrant': 'Team'
                    }

                    for old_name, new_name in column_mapping.items():
                        if old_name in entries_df.columns:
                            entries_df = entries_df.rename(
                                columns={old_name: new_name})

                    # Clean driver names
                    if 'Driver' in entries_df.columns:
                        entries_df['Driver'] = entries_df['Driver'].str.strip()
                    if 'Driver' in df.columns:
                        df['Driver'] = df['Driver'].str.strip()

                    # Merge team data if both Driver and Team
                    if 'Driver' in entries_df.columns and 'Team' in entries_df.columns:
                        df = df.merge(
                            entries_df[['Driver', 'Team']],
                            left_on='Driver', right_on='Driver', how='left'
                        )

                all_data.append(df)
        except Exception as e:
            print(f"Error processing {year_dir}: {e}")
            continue

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def create_target_using_f2_data(f3_df, f2_df):
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
                if not result or result in {'', 'DNS', 'WD', 'NC', 'EX', 'C'}:
                    continue
                participation += 1

            # For 2025, count any participation (>0 races)
            # For other years, use 50% threshold
            threshold = 0 if year == 2025 else total_races * 0.5

            f2_participation.append({
                'driver': driver,
                'year': year,
                'participation_ratio': participation / total_races,
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


def calculate_teammate_performance(df):
    """Calculate performance metrics relative to teammates."""
    if 'Team' not in df.columns:
        return df

    # Group by year and team to find teammates
    team_performance = []

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
            better_finishes = 0
            total_comparable = 0

            for race_col in race_cols:
                driver_result = str(driver_row[race_col]).strip()
                if not driver_result or driver_result in [
                        'DNS', 'WD', 'NC', 'EX', 'DSQ']:
                    continue

                try:
                    driver_pos = int(
                        driver_result.split()[0].replace(
                            '†', '').replace(
                            'Ret', '999'))
                except BaseException:
                    continue

                # Compare with each teammate
                for _, teammate_row in teammates_df.iterrows():
                    teammate_result = str(teammate_row[race_col]).strip()
                    if not teammate_result or teammate_result in [
                            'DNS', 'WD', 'NC', 'EX', 'DSQ']:
                        continue

                    try:
                        teammate_pos = int(
                            teammate_result.split()[0].replace(
                                '†', '').replace(
                                'Ret', '999'))

                        h2h_total += 1
                        total_comparable += 1

                        if driver_pos < teammate_pos:  # Lower pos number = better finish
                            h2h_wins += 1
                            better_finishes += 1

                    except BaseException:
                        continue

            # Calculate metrics
            h2h_rate = h2h_wins / h2h_total if h2h_total > 0 else 0.5

            # Calculate average position difference
            pos_differences = []
            for race_col in race_cols:
                driver_result = str(driver_row[race_col]).strip()
                if not driver_result or driver_result in [
                        'DNS', 'WD', 'NC', 'EX', 'DSQ']:
                    continue

                try:
                    driver_pos = int(
                        driver_result.split()[0].replace(
                            '†', '').replace(
                            'Ret', '25'))
                except BaseException:
                    continue

                teammate_positions = []
                for _, teammate_row in teammates_df.iterrows():
                    teammate_result = str(teammate_row[race_col]).strip()
                    if not teammate_result or teammate_result in [
                            'DNS', 'WD', 'NC', 'EX', 'DSQ']:
                        continue

                    try:
                        teammate_pos = int(
                            teammate_result.split()[0].replace(
                                '†', '').replace(
                                'Ret', '25'))
                        teammate_positions.append(teammate_pos)
                    except BaseException:
                        continue

                if teammate_positions:
                    avg_teammate_pos = np.mean(teammate_positions)
                    pos_differences.append(
                        avg_teammate_pos - driver_pos)  # Positive = driver better

            avg_pos_difference = np.mean(
                pos_differences) if pos_differences else 0

            team_performance.append({
                'Driver': driver,
                'year': year,
                'Team': team,
                'teammate_h2h_rate': h2h_rate,
                'avg_pos_vs_teammates': avg_pos_difference,
                'teammate_battles': h2h_total
            })

    # Convert to DataFrame and merge with original
    team_perf_df = pd.DataFrame(team_performance)
    if not team_perf_df.empty:
        df = df.merge(team_perf_df[['Driver',
                                    'year',
                                    'teammate_h2h_rate',
                                    'avg_pos_vs_teammates',
                                    'teammate_battles']],
                      on=['Driver',
                          'year'],
                      how='left')

        # Fill NaN values for drivers without teammates
        df['teammate_h2h_rate'] = df['teammate_h2h_rate'].fillna(0.5)
        df['avg_pos_vs_teammates'] = df['avg_pos_vs_teammates'].fillna(0)
        df['teammate_battles'] = df['teammate_battles'].fillna(0)
    else:
        df['teammate_h2h_rate'] = 0.5
        df['avg_pos_vs_teammates'] = 0
        df['teammate_battles'] = 0

    return df


def engineer_features(df):
    """Create features for ML model with F3 European consideration."""
    if df.empty:
        return pd.DataFrame()

    features_df = pd.DataFrame()
    features_df['year'] = df['year']
    features_df['driver'] = df['Driver']
    features_df['final_position'] = df['Pos'].astype(
        str).str.extract(r'(\d+)').astype(float)
    features_df['points'] = pd.to_numeric(
        df['Points'], errors='coerce').fillna(0)
    features_df['years_in_f3'] = df['years_in_f3']
    features_df['series_type'] = df.get('series_type', 'Unknown')
    features_df['is_f3_european'] = (
        features_df['series_type'] == 'F3_European').astype(int)

    race_cols = get_race_columns(df)
    wins, podiums, points_finishes, dnfs, races_completed = [], [], [], [], []

    # Calculate championship competitiveness (position relative to field size)
    field_sizes = []

    for _, row in df.iterrows():
        driver_wins = 0
        driver_podiums = 0
        driver_points = 0
        driver_dnfs = 0
        driver_races = 0

        # Calculate field size for this year/series
        year_series_data = df[
            (df['year'] == row['year']) &
            (df.get('series_type', 'F3_Main') == row.get('series_type', 'F3_Main'))
        ]
        field_size = len(year_series_data)
        field_sizes.append(field_size)

        for col in race_cols:
            if col not in row or pd.isna(row[col]):
                continue

            result = str(row[col]).strip()
            if not result:
                continue

            if any(x in result for x in ['Ret', 'DNS', 'WD', 'NC', 'DSQ']):
                driver_dnfs += 1
                driver_races += 1
                continue

            try:
                pos = int(
                    result.split()[0].replace(
                        '†',
                        '').replace(
                        'F',
                        '').replace(
                        'P',
                        ''))
                driver_races += 1

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

        wins.append(driver_wins)
        podiums.append(driver_podiums)
        points_finishes.append(driver_points)
        dnfs.append(driver_dnfs)
        races_completed.append(driver_races if driver_races > 0 else 1)

    features_df['wins'] = wins
    features_df['podiums'] = podiums
    features_df['points_finishes'] = points_finishes
    features_df['dnfs'] = dnfs
    features_df['races_completed'] = races_completed
    features_df['field_size'] = field_sizes

    # Teammate performance metrics
    features_df['teammate_h2h_rate'] = df.get('teammate_h2h_rate', 0.5)
    features_df['avg_pos_vs_teammates'] = df.get('avg_pos_vs_teammates', 0)
    features_df['teammate_battles'] = df.get('teammate_battles', 0)

    features_df = add_driver_features(features_df, f3_df)

    features_df['win_rate'] = features_df['wins'] / \
        features_df['races_completed']
    features_df['podium_rate'] = features_df['podiums'] / \
        features_df['races_completed']
    features_df['dnf_rate'] = features_df['dnfs'] / \
        features_df['races_completed']
    features_df['points_per_race'] = features_df['points'] / \
        features_df['races_completed']
    features_df['top_10_rate'] = features_df['points_finishes'] / \
        features_df['races_completed']

    return features_df


def calculate_years_in_f3_combined(df):
    """Calculate years in F3"""
    df = df.sort_values(by=['Driver', 'year'])

    # Group consecutive years for each driver
    df['years_in_f3'] = 0

    for driver in df['Driver'].unique():
        driver_data = df[df['Driver'] == driver].copy()

        # Calculate cumulative years, treating F3 and F3 European as continuous
        years_count = 0
        prev_year = None

        for idx, row in driver_data.iterrows():
            current_year = row['year']

            if prev_year is None or current_year == prev_year + 1:
                years_count += 1

            df.loc[idx, 'years_in_f3'] = years_count
            prev_year = current_year

    return df


def add_driver_features(features_df, f3_df):
    """Add age and academy features from cached JSON profiles."""
    profiles_dir = "driver_profiles"

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
                        profiles[driver] = json.load(f)
                except BaseException:
                    profiles[driver] = {'dob': None, 'academy': None}
            else:
                profiles[driver] = {'dob': None, 'academy': None}

    # Add features
    ages = []
    has_academy = []

    for _, row in features_df.iterrows():
        driver = row['driver']
        year = row['year']

        profile = profiles.get(driver, {})

        age = calculate_age(profile.get('dob'), year)
        ages.append(age)

        academy = profile.get('academy')
        has_academy.append(1 if academy else 0)

    features_df['age'] = ages
    features_df['has_academy'] = has_academy

    return features_df


def train_models(df):
    """Combined training function for all model types with proper cross-validation."""
    if df.empty:
        print("No data available for training")
        return {}, {}, None, None, None

    df_clean = df.dropna(subset=['moved_to_f2', 'final_position', 'points'])
    feature_cols = [
        'final_position', 'win_rate', 'podium_rate',
        'dnf_rate', 'points_per_race', 'top_10_rate',
        'years_in_f3', 'age', 'has_academy',
        'teammate_h2h_rate', 'avg_pos_vs_teammates',
    ]

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['moved_to_f2']

    print(f"Dataset size: {len(X)} drivers")
    print(f"F2 progressions: {y.sum()} ({y.mean():.2%})")

    # Scale features for neural networks
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69, stratify=y
    )
    X_train_scaled, X_test_scaled, _, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=69, stratify=y
    )

    # Traditional ML pipelines
    traditional_pipelines = {
        'Random Forest': ImbPipeline([
            ('smote', SMOTE(random_state=69)),
            ('classifier', RandomForestClassifier(
                random_state=69, class_weight='balanced'))
        ]),
        'Logistic Regression': ImbPipeline([
            ('smote', SMOTE(random_state=69)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=69, class_weight='balanced', max_iter=10000))
        ]),
        'XGBoost': ImbPipeline([
            ('smote', SMOTE(random_state=69)),
            ('classifier', xgb.XGBClassifier(
                random_state=69, eval_metric='logloss',
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            ))
        ]),
        'MLP': ImbPipeline([
            ('smote', SMOTE(random_state=69)),
            ('classifier', MLPClassifier(random_state=69, max_iter=10000))
        ])
    }

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
    scoring = ['roc_auc', 'precision', 'recall', 'f1']

    traditional_results = {}
    deep_results = {}

    # Train traditional models
    print("\n" + "=" * 50)
    print("TRAINING TRADITIONAL MODELS")
    print("=" * 50)

    for name, pipeline in traditional_pipelines.items():
        print(f"\n{name} Cross-Validation Results:")
        print("-" * 40)

        # Cross-validation
        cv_scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=-1
        )

        for metric in scoring:
            scores = cv_scores[f'test_{metric}']
            print(
                f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        # Fit and evaluate on test set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        probas_test = pipeline.predict_proba(X_test)[:, 1]

        # Calibration
        df_calib = pd.DataFrame(
            {'proba': probas_test, 'true': y_test.reset_index(drop=True)})
        bins = np.linspace(0.0, 1.0, 11)
        df_calib['bin'] = pd.cut(
            df_calib['proba'],
            bins=bins,
            include_lowest=True)
        bin_stats = df_calib.groupby('bin', observed=False)['true'].mean()
        pipeline.calibration_map = {
            interval: rate for interval,
            rate in bin_stats.items()}
        pipeline.calibration_bins = bins

        print("\nTest Set Results:")
        print(classification_report(y_test, y_pred))

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
    dnn_model = create_deep_nn_model(X_train_scaled.shape[1])

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5)
    ]

    dnn_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100, batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks, verbose=0
    )

    # Evaluate and calibrate
    raw_probas_dnn = dnn_model.predict(X_test_scaled, verbose=0).flatten()
    df_calib_dnn = pd.DataFrame(
        {'proba': raw_probas_dnn, 'true': y_test.reset_index(drop=True)})
    df_calib_dnn['bin'] = pd.cut(
        df_calib_dnn['proba'],
        bins=bins,
        include_lowest=True)
    bin_stats_dnn = df_calib_dnn.groupby('bin', observed=False)['true'].mean()
    dnn_model.calibration_map = {
        interval: rate for interval,
        rate in bin_stats_dnn.items()}
    dnn_model.calibration_bins = bins

    dnn_pred = (raw_probas_dnn > 0.5).astype(int)
    print("Keras DNN Classification Report:")
    print(classification_report(y_test, dnn_pred))
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
            torch.save(pytorch_model.state_dict(), 'pytorch_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    # Load best model and evaluate
    pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
    pytorch_model.eval()

    with torch.no_grad():
        logits = pytorch_model(X_test_torch)
        raw_probas_torch = torch.sigmoid(logits).cpu().numpy().flatten()

    # Calibrate PyTorch model
    df_calib_torch = pd.DataFrame(
        {'proba': raw_probas_torch, 'true': y_test.reset_index(drop=True)})
    df_calib_torch['bin'] = pd.cut(
        df_calib_torch['proba'],
        bins=bins,
        include_lowest=True)
    bin_stats_torch = df_calib_torch.groupby(
        'bin', observed=False)['true'].mean()
    pytorch_model.calibration_map = {
        interval: rate for interval,
        rate in bin_stats_torch.items()}
    pytorch_model.calibration_bins = bins

    pytorch_pred = (raw_probas_torch > 0.5).astype(int)
    print("PyTorch Classification Report:")
    print(classification_report(y_test, pytorch_pred))
    deep_results['PyTorch'] = pytorch_model

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Traditional Models: {list(traditional_results.keys())}")
    print(f"Deep Learning Models: {list(deep_results.keys())}")

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
        return

    X_current = current_df[feature_cols].fillna(0)

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
                    if name == 'Deep_NN':  # Keras model
                        raw_probas = model.predict(
                            X_processed, verbose=0).flatten()
                    else:  # PyTorch model
                        model.eval()
                        with torch.no_grad():
                            X_torch = torch.FloatTensor(X_processed)
                            if torch.cuda.is_available():
                                X_torch = X_torch.cuda()
                            logits = model(X_torch)
                            raw_probas = torch.sigmoid(
                                logits).cpu().numpy().flatten()
                else:  # Traditional models
                    raw_probas = model.predict_proba(X_processed)[:, 1]

                # Apply calibration
                bins = model.calibration_bins
                calib_map = model.calibration_map
                empirical_pct = []
                for p in raw_probas:
                    bin_interval = pd.cut(
                        [p], bins=bins, include_lowest=True)[0]
                    empirical_pct.append(
                        calib_map.get(
                            bin_interval, p) * 100.0)

                # Create results DataFrame
                results = pd.DataFrame({
                    'Driver': current_df['driver'],
                    'Position': current_df['final_position'],
                    'Points': current_df['points'],
                    'Win %': current_df['win_rate'],
                    'Podium %': current_df['podium_rate'],
                    'Top 10 %': current_df['top_10_rate'],
                    'DNF %': current_df['dnf_rate'],
                    'Points / Races': current_df['points_per_race'],
                    'Experience': current_df['years_in_f3'],
                    'Age': current_df['age'],
                    'Academy': current_df['has_academy'],
                    'H2H_Rate': current_df['teammate_h2h_rate'],
                    'Avg_Pos_Diff': current_df['avg_pos_vs_teammates'],
                    'Raw_Probability': raw_probas,
                    'Empirical_%': empirical_pct,
                    'Prediction': (raw_probas > 0.5).astype(int)
                }).sort_values('Empirical_%', ascending=False)

                # Print results
                print(f"\n{name} Predictions:")
                print("-" * 50)
                print(
                    results.head(10).to_string(
                        index=False,
                        float_format='%.3f'))

            except Exception as e:
                print(f"Error with {name} model: {e}")
                continue

    return results


# Deep Neural Network with TensorFlow/Keras

def create_deep_nn_model(
        input_dim,
        hidden_layers=[
            128,
            64,
            32],
        dropout_rate=0.3):
    """Create a deep neural network for F2 progression prediction."""
    model = Sequential()

    # Input layer (fixed warning)
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_layers[0], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

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


print("Loading F3 data...")
f3_df = load_and_combine_data('F3')

print("Loading F2 data...")
f2_df = load_and_combine_data('F2')

if f3_df.empty or f2_df.empty:
    print("No F2/F3 data found. Check file paths.")
    exit()

print("Calculating teammate performance metrics...")
f3_df = calculate_teammate_performance(f3_df)

print("Calculating years in F3 for each driver...")
f3_df = f3_df.sort_values(by=['Driver', 'year'])
f3_df['years_in_f3'] = f3_df.groupby('Driver').cumcount() + 1

print("Creating target variable based on F2 participation...")
f3_df = create_target_using_f2_data(f3_df, f2_df)

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
