import numpy as np
import os
import pandas as pd
import random
import re
import torch.optim as optim
import torch.nn as nn
import torch

from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from app.config import CURRENT_YEAR, NOT_PARTICIPATED_CODES, RETIREMENT_CODES, SEED
from app.core.loader import load_data, load_qualifying_data, load_standings_data
from app.core.utils import calculate_age, extract_position, get_race_columns
from app.core.pytorch_model import RacingPredictor

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch._dynamo.disable()
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_points_system(year):
    """Return points system parameters for a given year."""
    if year <= 2011:
        return {
            'feature_max': 12,  # 10 + 2 pole
            'sprint_max': 6,
            'feature_positions': [10, 8, 6, 5, 4, 3, 2, 1],
            'sprint_positions': [6, 5, 4, 3, 2, 1],
        }
    elif year <= 2020:
        return {
            'feature_max': 31,  # 25 + 4 pole + 2 FL
            'sprint_max': 17,   # 15 + 2 FL
            'feature_positions': [25, 18, 15, 12, 10, 8, 6, 4, 2, 1],
            'sprint_positions': [15, 12, 10, 8, 6, 4, 2, 1],
        }
    elif year == 2021:
        return {
            'race12_max': 17,   # 15 + 2 FL each
            'race3_max': 31,    # 25 + 4 pole + 2 FL
            'race12_positions': [15, 12, 10, 8, 6, 5, 4, 3, 2, 1],
            'race3_positions': [25, 18, 15, 12, 10, 8, 6, 4, 2, 1],
        }
    else:  # 2022-2025
        return {
            'feature_max': 28,  # 25 + 2 pole + 1 FL
            'sprint_max': 11,   # 10 + 1 FL
            'feature_positions': [25, 18, 15, 12, 10, 8, 6, 4, 2, 1],
            'sprint_positions': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        }


def identify_race_type(col_name, year):
    """Identify if column is sprint or feature race."""
    col_lower = col_name.lower()

    if year == 2021:
        # Triple header year - need different logic
        return 'race3' if 'r3' in col_lower else 'race12'
    elif year >= 2022:
        return 'sprint' if 'sr' in col_lower else 'feature'
    elif year <= 2020:
        # R1/FR = Feature, R2/SR = Sprint for 2010-2020
        return 'feature' if 'r1' in col_lower or 'fr' in col_lower else 'sprint'
    else:
        print(f'{col_name}, {year}')
        return None


def calculate_participation_stats(df, race_cols):
    """Calculate participation statistics for a dataframe."""
    stats = []

    for _, row in df.iterrows():
        # Vectorize race result checks
        race_results = [str(row[col]).strip() for col in race_cols]

        # Single pass through results
        participated_races = 0
        positions = []

        for result in race_results:
            if not result or result in NOT_PARTICIPATED_CODES:
                continue

            participated_races += 1

            if not any(x in result for x in RETIREMENT_CODES):
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

    race_cols = get_race_columns(df)
    if not race_cols:
        return df

    # Extract positions for all drivers at once (vectorized)
    position_matrix = np.full((len(df), len(race_cols)), np.nan)
    for i, col in enumerate(race_cols):
        position_matrix[:, i] = df[col].apply(
            lambda x: extract_position(str(x).strip()) if pd.notna(x) else np.nan
        ).values

    # Add positions to df temporarily
    df['_positions_matrix'] = [row for row in position_matrix]

    team_performance = []

    # Group once
    grouped = df.groupby(['year', 'Team'])

    for (year, team), team_df in grouped:
        if len(team_df) < 2:
            continue

        driver_indices = team_df.index.tolist()
        driver_names = team_df['Driver'].tolist()
        positions_list = team_df['_positions_matrix'].tolist()

        # Convert to numpy array for vectorized operations
        team_positions = np.array(positions_list)  # shape: (n_drivers, n_races)

        # Calculate pairwise comparisons
        n_drivers = len(driver_names)
        h2h_rates = np.full((n_drivers, n_drivers), np.nan)

        for i in range(n_drivers):
            for j in range(i + 1, n_drivers):
                pos_i = team_positions[i]
                pos_j = team_positions[j]

                # Valid races where both participated
                valid_mask = ~(np.isnan(pos_i) | np.isnan(pos_j))
                valid_count = valid_mask.sum()

                if valid_count > 0:
                    wins_i = (pos_i[valid_mask] < pos_j[valid_mask]).sum()
                    wins_j = (pos_j[valid_mask] < pos_i[valid_mask]).sum()
                    h2h_rates[i, j] = wins_i / valid_count
                    h2h_rates[j, i] = wins_j / valid_count

        # Calculate overall H2H rate for each driver
        for idx, (driver_idx, driver_name) in enumerate(zip(driver_indices, driver_names)):
            # Get all valid comparisons for this driver
            other_rates = h2h_rates[idx]
            valid_others = ~np.isnan(other_rates)

            if valid_others.any():
                # Weight by number of valid races against each teammate
                total_wins = 0
                total_races = 0

                for j, is_valid in enumerate(valid_others):
                    if is_valid and j != idx:
                        # Count valid races against teammate j
                        pos_self = team_positions[idx]
                        pos_other = team_positions[j]
                        valid_races = (~(np.isnan(pos_self) | np.isnan(pos_other))).sum()

                        total_races += valid_races
                        total_wins += other_rates[j] * valid_races

                h2h_rate = total_wins / total_races if total_races > 0 else 0.5
            else:
                h2h_rate = 0.5

            is_multi_team = df.loc[driver_idx].get('team_count', 1) > 1

            team_performance.append({
                'Driver': driver_name,
                'year': year,
                'Team': team,
                'teammate_h2h_rate': h2h_rate,
                'is_multi_team': is_multi_team
            })

    # Clean up
    df = df.drop('_positions_matrix', axis=1)

    # Convert to DataFrame and merge with original
    if team_performance:
        team_perf_df = pd.DataFrame(team_performance)
        df = df.merge(
            team_perf_df[['Driver', 'year', 'teammate_h2h_rate', 'is_multi_team']],
            on=['Driver', 'year'], how='left'
        )

    # Fill defaults
    df['teammate_h2h_rate'] = df['teammate_h2h_rate'].fillna(0.5)
    df['is_multi_team'] = df['is_multi_team'].fillna(False)

    return df


def calculate_qualifying_features(df, qualifying_df):
    """Calculate qualifying statistics for each driver-year combination."""
    position_columns = ['Pos.', 'Grid']

    # Vectorized position extraction function
    def extract_position_from_row(row):
        for col in position_columns:
            if col not in row.index or pd.isna(row[col]):
                continue

            # Handle numeric values directly
            if isinstance(row[col], (int, float)):
                return int(row[col])

            # String processing
            str_value = str(row[col]).strip()
            if str_value not in NOT_PARTICIPATED_CODES:
                match = re.search(r'\b\d{1,2}\b', str_value)
                if match:
                    return int(match.group())
        return np.nan

    # Extract positions for all rows at once
    qualifying_df['_extracted_pos'] = qualifying_df.apply(
        extract_position_from_row, axis=1
    )

    # Group and aggregate in one operation
    qualifying_stats = qualifying_df.groupby(['Driver', 'year'])['_extracted_pos'].agg([
        ('avg_quali_pos', lambda x: x.mean() if x.notna().any() else np.nan),
        ('std_quali_pos', lambda x: x.std() if x.notna().any() else np.nan)
    ]).reset_index()

    # Merge with main data
    if not qualifying_stats.empty:
        df = df.merge(qualifying_stats, on=['Driver', 'year'], how='left')

    # Fill missing values
    df[['avg_quali_pos', 'std_quali_pos']] = df.get(['avg_quali_pos', 'std_quali_pos'], np.nan)

    return df


def engineer_features(df):
    """Create features for ML models with race type separation."""
    if df.empty:
        return pd.DataFrame()

    df = calculate_teammate_performance(df)
    df = calculate_age(df)
    df = df.sort_values(by=['Driver', 'year'])
    df['experience'] = df.groupby('Driver').cumcount()

    features_df = pd.DataFrame({
        'year': df['year'],
        'Driver': df['Driver'],
        'series': df['series'],
        'dob': df['dob'],
        'nationality': df['nationality'],
        'pos': pd.to_numeric(df['Pos'], errors='coerce').fillna(-1).astype(int),
        'points': pd.to_numeric(df['Points'], errors='coerce').fillna(0),
        'experience': df['experience'],
        'age': df.get('age', np.nan),
        'team': df.get('Team'),
        'team_pos': df.get('team_pos', np.nan),
        'team_points': df.get('team_points', 0),
        'teammate_h2h_rate': df.get('teammate_h2h_rate', 0.5),
        'avg_quali_pos': df.get('avg_quali_pos', 0),
        'std_quali_pos': df.get('std_quali_pos', 0),
    })

    # Calculate race statistics
    race_stats = []
    cache_key_to_data = {}

    for _, row in df.iterrows():
        cache_key = (row['year'], row.get('series', 'F3'))
        if cache_key not in cache_key_to_data:
            year_series_data = df[
                (df['year'] == row['year']) &
                (df.get('series', 'F3') == row.get('series', 'F3'))
            ]
            race_cols = get_race_columns(year_series_data)
            cache_key_to_data[cache_key] = (race_cols)

        race_cols = cache_key_to_data[cache_key]
        points_system = get_points_system(row['year'])

        stats = {
            'sprint_points': 0,
            'feature_points': 0,
            'sprint_races': 0,
            'feature_races': 0,
            'sprint_wins': 0,
            'feature_wins': 0,
            'sprint_podiums': 0,
            'feature_podiums': 0,
            'sprint_point_finishes': 0,
            'feature_point_finishes': 0,
            'dnfs': 0,
            'finish_positions': [],
        }

        for col in race_cols:
            if col not in row or pd.isna(row[col]):
                continue

            result = str(row[col]).strip()
            if not result or result in NOT_PARTICIPATED_CODES:
                continue

            race_type = identify_race_type(col, row['year'])

            if any(x in result for x in RETIREMENT_CODES):
                if result != 'NC':
                    stats['dnfs'] += 1
                continue

            pos = extract_position(result)
            if not pos:
                continue

            stats['finish_positions'].append(pos)
            if row['year'] == 2021:
                if race_type == 'race3':
                    stats['feature_races'] += 1
                    if pos <= len(points_system['race3_positions']):
                        stats['feature_points'] += points_system['race3_positions'][pos - 1]
                        stats['feature_point_finishes'] += 1
                    if pos == 1:
                        stats['feature_wins'] += 1
                    if pos <= 3:
                        stats['feature_podiums'] += 1
                else:  # race12
                    stats['sprint_races'] += 1
                    if pos <= len(points_system['race12_positions']):
                        stats['sprint_points'] += points_system['race12_positions'][pos - 1]
                        stats['sprint_point_finishes'] += 1
                    if pos == 1:
                        stats['sprint_wins'] += 1
                    if pos <= 3:
                        stats['sprint_podiums'] += 1
            else:
                positions = points_system.get(f'{race_type}_positions', [])
                if race_type == 'sprint':
                    stats['sprint_races'] += 1
                    if pos <= len(positions):
                        stats['sprint_points'] += positions[pos - 1]
                        stats['sprint_point_finishes'] += 1
                    if pos == 1:
                        stats['sprint_wins'] += 1
                    if pos <= 3:
                        stats['sprint_podiums'] += 1
                else:  # feature
                    stats['feature_races'] += 1
                    if pos <= len(positions):
                        stats['feature_points'] += positions[pos - 1]
                        stats['feature_point_finishes'] += 1
                    if pos == 1:
                        stats['feature_wins'] += 1
                    if pos <= 3:
                        stats['feature_podiums'] += 1

        stats['races_completed'] = stats['feature_races'] + stats['sprint_races']
        stats['participation_rate'] = stats['races_completed'] / len(race_cols) if race_cols else 0
        stats['avg_finish_pos'] = np.mean(stats['finish_positions']) if stats['finish_positions'] else np.nan # noqa: 501
        stats['std_finish_pos'] = np.std(stats['finish_positions']) if stats['finish_positions'] else np.nan # noqa: 501

        race_stats.append(stats)

    # Add race statistics
    for stat_name in ['sprint_points', 'feature_points', 'sprint_races', 'feature_races',
                      'sprint_wins', 'feature_wins', 'sprint_podiums', 'feature_podiums',
                      'sprint_point_finishes', 'feature_point_finishes',
                      'dnfs', 'races_completed', 'participation_rate']:
        features_df[stat_name] = [stats[stat_name] for stats in race_stats]

    features_df = features_df[features_df['races_completed'] > 0]

    # Race-type specific rates
    features_df['sprint_races'] = features_df['sprint_races'].fillna(0)
    features_df['feature_races'] = features_df['feature_races'].fillna(0)

    # Win rates by race type
    features_df['sprint_win_rate'] = np.where(
        features_df['sprint_races'] > 0,
        features_df['sprint_wins'].fillna(0) / features_df['sprint_races'],
        0
    )
    features_df['feature_win_rate'] = np.where(
        features_df['feature_races'] > 0,
        features_df['feature_wins'].fillna(0) / features_df['feature_races'],
        0
    )

    features_df['wins'] = features_df['feature_wins'] + features_df['sprint_wins']
    features_df['podiums'] = features_df['feature_podiums'] + features_df['sprint_podiums']

    # Overall rates
    features_df['win_rate'] = features_df['wins'] / features_df['races_completed']
    features_df['dnf_rate'] = features_df['dnfs'] / features_df['races_completed']

    # Championship position percentile
    features_df['champ_pos_pct'] = features_df.groupby('year')['pos'].rank(pct=True)

    # Target encode nationality
    if 'promoted' in df.columns:
        global_mean = df['promoted'].mean()
        nationality_stats = df.groupby('nationality').agg({
            'promoted': ['sum', 'count']
        }).droplevel(0, axis=1)
        alpha = 10
        nationality_stats['smoothed_rate'] = (
            (nationality_stats['sum'] + alpha * global_mean) /
            (nationality_stats['count'] + alpha)
        )
        features_df['nationality_encoded'] = features_df['nationality'].map(
            nationality_stats['smoothed_rate']
        ).fillna(global_mean)
    else:
        features_df['nationality_encoded'] = 0.2

    return features_df


def create_target_variable(feeder_df, parent_df, series):
    """Create target variable for parent series participation."""
    if feeder_df.empty or parent_df.empty:
        feeder_df['promoted'] = np.nan
        return feeder_df

    feeder_df['promoted'] = 0
    max_parent_year = parent_df['year'].max()

    # Get last feeder season per driver
    last_feeder_seasons = feeder_df.groupby('Driver')['year'].max()

    # Build participation lookup
    participation_lookup = {}
    for year, year_df in parent_df.groupby('year'):
        race_cols = get_race_columns(year_df)
        if not race_cols:
            continue

        threshold = 0 if year == CURRENT_YEAR else len(race_cols) * 0.4
        stats = calculate_participation_stats(year_df, race_cols)

        for stat in stats:
            key = (stat['Driver'], year)
            participation_lookup[key] = stat['participated_races'] > threshold

    # Target assignment
    years_to_check = [1, 2, 3] if series == 'F1' else [1]

    def check_promotion(row):
        driver = row['Driver']
        year = row['year']
        last_year = last_feeder_seasons.get(driver)

        # Only process last feeder season
        if year != last_year:
            return 0

        # Can't observe future
        if year + 1 > max_parent_year:
            return np.nan

        # Check future years
        for offset in years_to_check:
            target_year = year + offset
            if target_year > max_parent_year:
                break
            if participation_lookup.get((driver, target_year), False):
                return 1
        return 0

    feeder_df['promoted'] = feeder_df.apply(check_promotion, axis=1)
    return feeder_df


def train_models(df):
    """Training function."""
    if df.empty:
        print("No data available for training")
        return {}, None, None

    df_clean = df.dropna(subset=['promoted'])

    if df_clean['series'][0] == 'F2':
        feature_cols = [
            'experience', 'std_quali_pos', 'feature_win_rate',
            'champ_pos_pct', 'nationality_encoded'
        ]
    else:
        feature_cols = [
            'avg_quali_pos', 'sprint_win_rate',
            'feature_win_rate', 'experience',
            'teammate_h2h_rate', 'nationality_encoded',
            'participation_rate', 'dnf_rate', 'champ_pos_pct',
        ]

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['promoted']
    years = df_clean['year']

    # Temporal split
    unique_years = sorted(years.unique())
    n_train_years = int(len(unique_years) * 0.8)
    train_years = unique_years[:n_train_years]

    train_mask = years.isin(train_years)
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(X_train, y_train))

    X_train_sub, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_sub, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    print(f"Training subset: {len(X_train_sub)} samples, {y_train_sub.sum()} promotions ({y_train_sub.mean():.2%})") # noqa: 501
    print(f"Validation: {len(X_val)} samples, {y_val.sum()} promotions ({y_val.mean():.2%})")
    print(f"Test: {len(X_test)} samples, {y_test.sum()} promotions ({y_test.mean():.2%})")

    # Traditional ML pipelines
    traditional_pipelines = {
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_jobs=-1))
        ]),
        'LightGBM': Pipeline([
            ('classifier', LGBMClassifier(
                random_state=SEED,
                class_weight='balanced',
                verbosity=-1,
                n_jobs=-1
            ))
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=SEED,
                class_weight='balanced',
                max_iter=10000
            ))
        ]),
        'Naive Bayes': Pipeline([
            ('classifier', GaussianNB())
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                random_state=SEED,
                class_weight='balanced',
                probability=True
            ))
        ]),
        'Random Forest': Pipeline([
            ('classifier', RandomForestClassifier(
                random_state=SEED,
                class_weight='balanced_subsample'
            ))
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

        # Fit on training subset
        pipeline.fit(X_train_sub, y_train_sub)

        # Evaluate on validation set
        probas_val = pipeline.predict_proba(X_val)[:, 1]

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        probas_test = pipeline.predict_proba(X_test)[:, 1]

        # Calibration using validation set
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(probas_val, y_val)
        pipeline.calibrator = iso_reg

        print("\nTest Set Results:")
        print(classification_report(y_test, y_pred))
        calibrated_probas = pipeline.calibrator.transform(probas_test)
        pr_auc_calibrated = average_precision_score(y_test, calibrated_probas)
        print(f"Test PR-AUC (calibrated): {pr_auc_calibrated:.4f}")

        results[name] = pipeline

    # Train PyTorch Model
    print("\nTraining PyTorch Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Scale features
    scaler = RobustScaler()
    X_train_sub_scaled = scaler.fit_transform(X_train_sub)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_torch = torch.FloatTensor(X_train_sub_scaled).to(device)
    y_train_torch = torch.FloatTensor(y_train_sub.values).to(device)
    X_val_torch = torch.FloatTensor(X_val_scaled).to(device)
    y_val_torch = torch.FloatTensor(y_val.values).to(device)
    X_test_torch = torch.FloatTensor(X_test_scaled).to(device)

    pytorch_model = RacingPredictor(X_train_sub_scaled.shape[1]).to(device)

    # Calculate class weights for PyTorch
    n_neg = (y_train_sub == 0).sum()
    n_pos = (y_train_sub == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(pytorch_model.parameters(), lr=0.01, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    # Training loop with proper validation
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(30):
        pytorch_model.train()
        optimizer.zero_grad()

        outputs = pytorch_model(X_train_torch).squeeze()
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()

        # Validation
        pytorch_model.eval()
        with torch.no_grad():
            val_outputs = pytorch_model(X_val_torch).squeeze()
            val_loss = criterion(val_outputs, y_val_torch)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = pytorch_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 3:
                break

    # Load best model and evaluate
    pytorch_model.load_state_dict(best_state)
    pytorch_model.eval()

    # Validation and Test evaluation
    with torch.no_grad():
        val_probas = torch.sigmoid(pytorch_model(X_val_torch)).cpu().numpy().flatten()
        test_probas = torch.sigmoid(pytorch_model(X_test_torch)).cpu().numpy().flatten()

    # Calibrate PyTorch model using validation set
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(val_probas, y_val)
    pytorch_model.calibrator = iso_reg

    calibrated_probas = pytorch_model.calibrator.transform(test_probas)
    y_pred = (test_probas > 0.5).astype(int)
    print("\nPyTorch Test Results:")
    print(classification_report(y_test, y_pred))

    pr_auc_calibrated = average_precision_score(y_test, calibrated_probas)
    print(f"Test PR-AUC (calibrated): {pr_auc_calibrated:.4f}")
    results['PyTorch'] = pytorch_model

    return results, feature_cols, scaler


def predict_drivers(models, df, feature_cols, scaler=None):
    """Make predictions for current year drivers"""
    current_df = df[df['year'] == CURRENT_YEAR].copy()
    if current_df.empty:
        current_df = df[df['year'] == df['year'].max()].copy()
    if current_df.empty:
        print("No current data found for predictions")
        return pd.DataFrame()

    X_current = current_df[feature_cols].fillna(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = None

    for name, model in models.items():
        try:
            # Get raw probabilities based on model type
            if name == 'PyTorch':
                if scaler is not None:
                    X_processed = scaler.transform(X_current)
                else:
                    X_processed = X_current
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
                'Driver': current_df['Driver'],
                'Nat.': current_df['nationality'],
                'Nat_encoded': current_df['nationality_encoded'],
                'Pos': current_df['pos'],
                'Points': current_df['points'],
                'Wins': current_df['wins'],
                'Podiums': current_df['podiums'],
                'Win %': current_df['win_rate'],
                'DNF %': current_df['dnf_rate'],
                'Participation %': current_df['participation_rate'],
                'Exp': current_df['experience'],
                'DoB': current_df['dob'],
                'Age': current_df['age'],
                'Teammate_h2h': current_df['teammate_h2h_rate'],
                'Team': current_df['team'],
                'Team Pos': current_df['team_pos'],
                'Team Points': current_df['team_points'],
                'Raw_Prob': raw_probas,
                'Empirical_%': empirical_pct
            }).sort_values('Empirical_%', ascending=False)

            print(f"\n{name} Predictions:")
            print("=" * 70)
            print(results.head(3).to_string(index=False, float_format='%.3f'))

        except Exception as e:
            print(f"Error with {name} model: {e}")
            continue

    if results is not None:
        return results
    return pd.DataFrame()


import cProfile  # noqa: 402
import pstats  # noqa: 402
import psutil  # noqa: 402


def main():
    """Wrap with profiling and memory measurements."""
    process = psutil.Process()

    # Record starting memory (RSS in bytes)
    mem_start = process.memory_info().rss

    # Set up profiler
    profiler = cProfile.Profile()
    profiler.enable()

    series = ['F3', 'F2']

    print(f"Loading {series[0]} qualifying data...")
    feeder_quali_data = load_qualifying_data(series[0])

    feeder_df = load_data(series[0])
    parent_df = load_standings_data(series[1], 'drivers')

    print("Adding qualifying features...")
    feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_data)

    print(f"Creating target variable based on {series[1]} participation...")
    feeder_df = create_target_variable(feeder_df, parent_df, series[1])

    print("Engineering features...")
    features_df = engineer_features(feeder_df)
    features_df['promoted'] = feeder_df['promoted']
    del feeder_df, parent_df, feeder_quali_data

    print("Training all models...")
    models, feature_cols, scaler = train_models(features_df)

    print(f"Making predictions for {series[0]} {CURRENT_YEAR} drivers...")
    predict_drivers(models, features_df, feature_cols, scaler)

    # Stop profiling
    profiler.disable()

    # Record ending memory
    mem_end = process.memory_info().rss

    # Print memory usage summary
    print(f"\nMemory (RSS) before: {mem_start / (1024**2):.2f} MiB")
    print(f"Memory (RSS) after: {mem_end   / (1024**2):.2f} MiB")
    print(f"Memory delta: {(mem_end - mem_start) / (1024**2):.2f} MiB\n")

    # Print top 5 functions by cumulative time
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(5)


if __name__ == "__main__":  # pragma: no cover
    main()
