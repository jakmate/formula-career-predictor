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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from app.config import CURRENT_YEAR, NOT_PARTICIPATED_CODES, RETIREMENT_CODES, SEED
from app.core.loader import load_data, load_qualifying_data, load_standings_data
from app.core.utils import calculate_age, extract_position, get_race_columns
from app.core.pytorch_model import RacingPredictor

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    race_cols = get_race_columns(df)
    if not race_cols:
        return df

    team_performance = []

    # Group by year and team to find teammates
    for (year, team), team_df in df.groupby(['year', 'Team']):
        if len(team_df) < 2:  # Skip teams with only one driver
            continue

        # Pre-extract positions for all drivers in this team
        def extract_positions_vectorized(row, race_cols):
            return [extract_position(str(row[col]).strip()) for col in race_cols]

        driver_positions = {
            row['Driver']: extract_positions_vectorized(row, race_cols)
            for _, row in team_df.iterrows()
        }

        # Calculate h2h for each driver pair once
        drivers = list(driver_positions.keys())
        h2h_results = {}

        for i in range(len(drivers)):
            for j in range(i + 1, len(drivers)):
                driver1, driver2 = drivers[i], drivers[j]
                pos1_list = driver_positions[driver1]
                pos2_list = driver_positions[driver2]

                wins1 = wins2 = total = 0
                for pos1, pos2 in zip(pos1_list, pos2_list):
                    if pos1 and pos2:
                        total += 1
                        if pos1 < pos2:
                            wins1 += 1
                        elif pos2 < pos1:
                            wins2 += 1

                if total > 0:
                    h2h_results[(driver1, driver2)] = wins1 / total
                    h2h_results[(driver2, driver1)] = wins2 / total

        # Build results for each driver
        for _, driver_row in team_df.iterrows():
            driver = driver_row['Driver']

            # Calculate overall h2h rate against all teammates
            total_wins = total_races = 0
            for other_driver in drivers:
                if other_driver != driver:
                    key = (driver, other_driver)
                    if key in h2h_results:
                        teammate_total = sum(1 for p1, p2 in zip(driver_positions[driver], driver_positions[other_driver]) if p1 and p2)  # noqa: 501
                        total_races += teammate_total
                        total_wins += h2h_results[key] * teammate_total

            h2h_win_rate = total_wins / total_races if total_races > 0 else 0.5
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


def calculate_qualifying_features(df, qualifying_df):
    """Calculate qualifying-based features for drivers."""
    if qualifying_df.empty:
        df['avg_quali_pos'] = np.nan
        return df

    # Calculate qualifying statistics for each driver-year combination
    qualifying_stats = []

    for (driver, year), driver_data in qualifying_df.groupby(['Driver', 'year']):
        positions = []
        # Define priority order for qualifying position columns
        position_columns = ['Pos.', 'Grid']  # Grid as backup

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
    df = calculate_age(df)
    df = df.sort_values(by=['Driver', 'year'])
    df['experience'] = df.groupby('Driver').cumcount()

    features_df = pd.DataFrame({
        'year': df['year'],
        'driver': df['Driver'],
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
        'avg_quali_pos': df.get('avg_quali_pos', np.nan),
    })

    # Calculate race statistics for each driver
    race_stats = []
    cache_key_to_data = {}

    for _, row in df.iterrows():
        # Create cache key for this year/series combination
        cache_key = (row['year'], row.get('series', 'F3'))

        # Cache race data
        if cache_key not in cache_key_to_data:
            year_series_data = df[
                (df['year'] == row['year']) &
                (df.get('series', 'F3') == row.get('series', 'F3'))
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
    for stat_name in ['wins', 'podiums', 'top_10s', 'dnfs',
                      'races_completed', 'participation_rate']:
        features_df[stat_name] = [stats[stat_name] for stats in race_stats]

    # Filter out drivers with no races
    features_df = features_df[features_df['races_completed'] > 0]

    features_df['win_rate'] = features_df['wins'] / features_df['races_completed']
    features_df['top_10_rate'] = features_df['top_10s'] / features_df['races_completed']
    features_df['dnf_rate'] = features_df['dnfs'] / features_df['races_completed']

    # Target encode nationality
    if 'promoted' in df.columns:
        global_mean = df['promoted'].mean()
        nationality_stats = df.groupby('nationality').agg({
            'promoted': ['sum', 'count']
        }).droplevel(0, axis=1)

        # Smoothing factor (higher = more conservative)
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

    features_df['era'] = np.where(df['year'] >= 2019, 1, 0)
    features_df['consistency_score'] = features_df['participation_rate'] * \
        (1 - features_df['dnf_rate'])

    return features_df


def create_target_variable(feeder_df, parent_df, series):
    """Create target variable for parent series participation."""
    if feeder_df.empty or parent_df.empty:
        feeder_df['promoted'] = np.nan
        return feeder_df

    # Initialize target column
    feeder_df['promoted'] = 0
    max_parent_year = parent_df['year'].max()

    # Get last feeder season for each driver
    last_feeder_seasons = feeder_df.groupby('Driver')['year'].max().reset_index()

    # Process parent data to determine participation
    parent_participation = []
    for year, year_df in parent_df.groupby('year'):
        race_cols = get_race_columns(year_df)
        if not race_cols:
            continue

        participation_stats = calculate_participation_stats(year_df, race_cols)
        threshold = 0 if year == CURRENT_YEAR else len(race_cols) * 0.4

        for stat in participation_stats:
            parent_participation.append({
                'driver': stat['Driver'],
                'year': year,
                'participated': stat['participated_races'] > threshold
            })

    parent_participation_df = pd.DataFrame(parent_participation)

    # Determine target values
    moved_drivers = {}
    for _, row in last_feeder_seasons.iterrows():
        driver = row['Driver']
        last_feeder_year = row['year']

        # Skip if we can't observe future seasons
        if last_feeder_year + 1 > max_parent_year:
            moved_drivers[(driver, last_feeder_year)] = np.nan
            continue

        years = []
        if series == 'F1':
            years = [1, 2, 3, 4, 5]
        else:
            years = [1]

        # Check next years for participation
        moved = 0
        for offset in years:
            target_year = last_feeder_year + offset
            if target_year > max_parent_year:
                break

            participation = parent_participation_df[
                (parent_participation_df['driver'] == driver) &
                (parent_participation_df['year'] == target_year)
            ]

            if not participation.empty and participation['participated'].iloc[0]:
                moved = 1
                break

        moved_drivers[(driver, last_feeder_year)] = moved

    # Apply target values
    for idx, row in feeder_df.iterrows():
        driver = row['Driver']
        year = row['year']
        if (driver, year) in moved_drivers:
            feeder_df.at[idx, 'promoted'] = moved_drivers[(driver, year)]

    return feeder_df


def train_models(df):
    """Training function."""
    if df.empty:
        print("No data available for training")
        return {}, None, None

    df_clean = df.dropna(subset=['promoted'])
    feature_cols = [
        'avg_quali_pos',
        'win_rate', 'top_10_rate',
        'experience', 'age',
        'teammate_h2h_rate',
        'nationality_encoded',
        'era',
        'consistency_score'
    ]

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['promoted']
    years = df_clean['year']

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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # For a single validation split, take first fold
    train_idx, val_idx = next(skf.split(X_train, y_train))

    X_train_sub = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_train_sub = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    print(f"Training subset: {len(X_train_sub)} samples, \
          {y_train_sub.sum()} promotions ({y_train_sub.mean():.2%})")
    print(f"Validation: {len(X_val)} samples, {y_val.sum()} promotions ({y_val.mean():.2%})")
    print(f"Test: {len(X_test)} samples, {y_test.sum()} promotions ({y_test.mean():.2%})")

    # Traditional ML pipelines
    traditional_pipelines = {
        'Random Forest': Pipeline([
            ('classifier', RandomForestClassifier(
                random_state=SEED,
                class_weight='balanced_subsample'
            ))
        ]),
        'Logistic Regression': Pipeline([
            ('classifier', LogisticRegression(
                random_state=SEED,
                class_weight='balanced',
                max_iter=10000
            ))
        ]),
        'LightGBM': Pipeline([
            ('classifier', LGBMClassifier(
                random_state=SEED,
                class_weight='balanced',
                verbosity=-1
            ))
        ]),
        'MLP': Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', MLPClassifier(
                random_state=SEED,
                max_iter=10000
            ))
        ]),
        'SVM': Pipeline([
            ('classifier', SVC(
                random_state=SEED,
                class_weight='balanced',
                probability=True
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
        pr_auc = average_precision_score(y_test, probas_test)
        print(f"Test Precision-Recall AUC: {pr_auc:.4f}")

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
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Training loop with proper validation
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):
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
            best_state_dict = pytorch_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    # Load best model and evaluate
    pytorch_model.load_state_dict(best_state_dict)
    pytorch_model.eval()

    # Validation evaluation
    with torch.no_grad():
        val_logits = pytorch_model(X_val_torch)
        val_probas_torch = torch.sigmoid(val_logits).cpu().numpy().flatten()

    # Test evaluation
    with torch.no_grad():
        test_logits = pytorch_model(X_test_torch)
        test_probas_torch = torch.sigmoid(test_logits).cpu().numpy().flatten()

    # Calibrate PyTorch model using validation set
    iso_reg_torch = IsotonicRegression(out_of_bounds='clip')
    iso_reg_torch.fit(val_probas_torch, y_val)
    pytorch_model.calibrator = iso_reg_torch

    pytorch_pred = (test_probas_torch > 0.5).astype(int)
    print("\nPyTorch Test Results:")
    print(classification_report(y_test, pytorch_pred))
    pr_auc_torch = average_precision_score(y_test, test_probas_torch)
    print(f"PyTorch Test Precision-Recall AUC: {pr_auc_torch:.4f}")
    results['PyTorch'] = pytorch_model

    return results, feature_cols, scaler


def predict_drivers(models, df, feature_cols, scaler=None):
    """Make predictions for current year drivers"""
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
                'nationality_encoded': current_df['nationality_encoded'],
                'Pos': current_df['pos'],
                'Avg Quali': current_df['avg_quali_pos'],
                'Points': current_df['points'],
                'Wins': current_df['wins'],
                'Podiums': current_df['podiums'],
                'Win %': current_df['win_rate'],
                'Top 10 %': current_df['top_10_rate'],
                'DNF %': current_df['dnf_rate'],
                'Participation %': current_df['participation_rate'],
                'Consistency': current_df['consistency_score'],
                'Exp': current_df['experience'],
                'DoB': current_df['dob'],
                'Age': current_df['age'],
                'teammate_h2h_rate': current_df['teammate_h2h_rate'],
                'team': current_df['team'],
                'team_pos': current_df['team_pos'],
                'team_points': current_df['team_points'],
                'Raw_Prob': raw_probas,
                'Empirical_%': empirical_pct
            }).sort_values('Empirical_%', ascending=False)

            print(f"\n{name} Predictions:")
            print("=" * 70)
            print(results.head(5).to_string(index=False, float_format='%.3f'))

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

    # Print top 20 functions by cumulative time
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)


if __name__ == "__main__":  # pragma: no cover
    main()
