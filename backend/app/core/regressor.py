import numpy as np
import os
import pandas as pd
import random
import re
import torch.optim as optim
import torch.nn as nn
import torch

from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from app.core.loader import load_data, load_qualifying_data

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
        clean_str = result_str.split()[0].replace('†', '')
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

    race_cols = get_race_columns(df)
    if not race_cols:
        return df

    team_performance = []

    # Group by year and team to find teammates
    for (year, team), team_df in df.groupby(['year', 'Team']):
        if len(team_df) < 2:  # Skip teams with only one driver
            continue

        # Pre-extract positions for all drivers in this team
        driver_positions = {}
        for _, row in team_df.iterrows():
            driver = row['Driver']
            positions = []
            for race_col in race_cols:
                pos = extract_position(str(row[race_col]).strip())
                positions.append(pos)
            driver_positions[driver] = positions

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
    df = calculate_age(df)
    df = df.sort_values(by=['Driver', 'year'])
    df['experience'] = df.groupby('Driver').cumcount()

    features_df = pd.DataFrame({
        'year': df['year'],
        'driver': df['Driver'],
        'dob': df['dob'],
        'nationality': df['nationality'],
        'pos': pd.to_numeric(df['Pos'], errors='coerce'),
        'points': pd.to_numeric(df['Points'], errors='coerce').fillna(0),
        'experience': df['experience'],
        'age': df.get('age', np.nan),
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

    features_df['win_rate'] = features_df['wins'] / features_df['races_completed']
    features_df['podium_rate'] = features_df['podiums'] / features_df['races_completed']
    features_df['top_10_rate'] = features_df['top_10s'] / features_df['races_completed']
    features_df['dnf_rate'] = features_df['dnfs'] / features_df['races_completed']
    features_df['points_share'] = features_df['points'] / (features_df['team_points'] + 1)

    return features_df


class RacingPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2):
        super(RacingPredictor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_target_variable(df):
    """Create target variable for final championship position prediction."""
    df['target_position'] = pd.to_numeric(df['Pos'], errors='coerce')
    return df


def get_races_remaining(df, current_year):
    """Calculate races remaining in the current season."""
    # Get race columns for current year
    current_df = df[df['year'] == current_year]
    if current_df.empty:
        return 0, 0

    race_cols = get_race_columns(current_df)
    total_races = len(race_cols)

    # Count completed races (any driver has a result)
    completed_races = 0
    for col in race_cols:
        if current_df[col].notna().any():
            completed_races += 1

    races_remaining = total_races - completed_races
    return races_remaining, total_races


def train_models(df):
    """Training function with temporal split for position prediction."""
    if df.empty:
        print("No data available for training")
        return {}, {}, None, None, None, None, None, None

    df_clean = df.dropna(subset=['target_position', 'pos'])
    feature_cols = [
        'avg_finish_pos', 'std_finish_pos',
        'avg_quali_pos', 'std_quali_pos',
        'win_rate', 'podium_rate', 'top_10_rate',
        'participation_rate', 'dnf_rate',
        'experience', 'age',
        'teammate_h2h_rate', 'points_share',
        'pole_rate', 'top_10_starts_rate',
        'races_completed'
    ]

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['target_position']
    years = df_clean['year']

    print(f"Dataset size: {len(X)} drivers")
    print(f"Position range: {y.min():.0f} - {y.max():.0f}")
    print(f"Year range: {years.min()} - {years.max()}")

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

    print(f"Training: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Regression models optimized for position prediction
    regression_models = {
        'Random Forest': RandomForestRegressor(random_state=SEED, n_estimators=200, max_depth=15),
        'Linear Regression': LinearRegression(),
        'LightGBM': LGBMRegressor(random_state=SEED, verbosity=-1, objective='regression'),
        'MLP': MLPRegressor(random_state=SEED, max_iter=1000, early_stopping=True),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
    }

    results = {}

    print("\n" + "=" * 50)
    print("TRAINING POSITION PREDICTION MODELS")
    print("=" * 50)

    # Train regression models
    for name, model in regression_models.items():
        print(f"\nTraining {name}:")
        print("-" * 40)

        # Use scaled features for models that need it
        if name in ['Linear Regression', 'SVR', 'MLP']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")

        results[name] = model

    # Train PyTorch Model
    print("\nTraining PyTorch Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to PyTorch tensors
    X_train_torch = torch.FloatTensor(X_train_scaled).to(device)
    y_train_torch = torch.FloatTensor(y_train.values).to(device)
    X_test_torch = torch.FloatTensor(X_test_scaled).to(device)

    pytorch_model = RacingPredictor(X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Validation split
    n_train_samples = len(X_train_torch)
    val_split_idx = int(n_train_samples * 0.8)

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
        predictions_torch = pytorch_model(X_test_torch).cpu().numpy().flatten()

    mse_torch = mean_squared_error(y_test, predictions_torch)
    mae_torch = mean_absolute_error(y_test, predictions_torch)
    r2_torch = r2_score(y_test, predictions_torch)

    print(f"PyTorch MSE: {mse_torch:.2f}")
    print(f"PyTorch MAE: {mae_torch:.2f}")
    print(f"PyTorch R²: {r2_torch:.4f}")

    results['PyTorch'] = pytorch_model

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return results, feature_cols, scaler


def predict_final_championship_standings(models, df, feature_cols, scaler=None):
    """Predict final championship positions for the entire grid."""
    current_year = CURRENT_YEAR
    current_df = df[df['year'] == current_year].copy()
    if current_df.empty:
        current_year = df['year'].max()
        current_df = df[df['year'] == current_year].copy()
    if current_df.empty:
        print("No current data found for predictions")
        return pd.DataFrame()

    # Get race information
    races_remaining, total_races = get_races_remaining(df, current_year)
    season_progress = (total_races - races_remaining) / total_races if total_races > 0 else 0

    print(f"\nSeason Progress: {season_progress:.1%}")
    print(f"Races completed: {total_races - races_remaining}/{total_races}")
    print(f"Races remaining: {races_remaining}")

    X_current = current_df[feature_cols].fillna(0)
    all_predictions = {}

    # Adjust predictions based on season progress
    confidence_factor = max(0.3, season_progress)

    for name, model in models.items():
        try:
            # Get predictions based on model type
            if name == 'PyTorch':
                if scaler is not None:
                    X_processed = scaler.transform(X_current)
                else:
                    X_processed = X_current
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_torch = torch.FloatTensor(X_processed).to(device)
                    predictions = model(X_torch).cpu().numpy().flatten()
            else:
                if name in ['Linear Regression', 'SVR', 'MLP']:
                    X_processed = scaler.transform(X_current) if scaler else X_current
                else:
                    X_processed = X_current
                predictions = model.predict(X_processed)

            # Constrain predictions to valid position range
            n_drivers = len(current_df)
            predictions = np.clip(predictions, 1, n_drivers)

            all_predictions[name] = predictions

        except Exception as e:
            print(f"Error with {name} model: {e}")
            continue

    if not all_predictions:
        return pd.DataFrame()

    # Create ensemble prediction (average of all models)
    ensemble_pred = np.mean(list(all_predictions.values()), axis=0)

    # Adjust predictions based on current position and season progress
    current_positions = current_df['pos'].values

    # Weight current position more heavily as season progresses
    adjusted_predictions = (
        ensemble_pred * (1 - confidence_factor) +
        current_positions * confidence_factor
    )

    # Create results DataFrame
    results = pd.DataFrame({
        'Driver': current_df['driver'],
        'Nat.': current_df['nationality'],
        'Current_Pos': current_df['pos'],
        'Current_Points': current_df['points'],
        'Predicted_Final_Pos': adjusted_predictions,
        'Position_Change': current_df['pos'] - adjusted_predictions,
        'Avg_Finish_Pos': current_df['avg_finish_pos'],
        'Win_Rate': current_df['win_rate'],
        'Podium_Rate': current_df['podium_rate'],
        'Age': current_df['age'],
        'Experience': current_df['experience'],
        'Team': current_df['team'],
        'Races_Completed': current_df['races_completed'],
        'Confidence': confidence_factor
    })

    # Add individual model predictions
    for name, preds in all_predictions.items():
        results[f'{name}_Pred'] = preds

    # Sort by predicted final position (best first)
    results = results.sort_values('Predicted_Final_Pos', ascending=True)

    # Add final ranking
    results['Predicted_Rank'] = range(1, len(results) + 1)

    print(f"\n{current_year} Final Championship Standings Prediction:")
    print("=" * 100)
    print(f"Prediction Confidence: {confidence_factor:.1%}")
    print("-" * 100)

    display_cols = ['Predicted_Rank', 'Driver', 'Nat.', 'Current_Pos', 'Predicted_Final_Pos',
                    'Position_Change', 'Current_Points', 'Win_Rate', 'Podium_Rate', 'Team']

    # Show full grid
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    print(results[display_cols].to_string(index=False, float_format='%.1f'))

    # Highlight biggest movers
    print("\nBiggest Predicted Gainers:")
    print("-" * 50)
    gainers = results[results['Position_Change'] > 0].nlargest(5, 'Position_Change')
    if not gainers.empty:
        for _, driver in gainers.iterrows():
            print(f"{driver['Driver']}: {driver['Current_Pos']:.0f} → {driver['Predicted_Final_Pos']:.1f} "  # noqa: 501
                  f"(+{driver['Position_Change']:.1f} positions)")
    else:
        print("No significant gainers predicted")

    print("\nBiggest Predicted Fallers:")
    print("-" * 50)
    fallers = results[results['Position_Change'] < 0].nsmallest(5, 'Position_Change')
    if not fallers.empty:
        for _, driver in fallers.iterrows():
            print(f"{driver['Driver']}: {driver['Current_Pos']:.0f} → {driver['Predicted_Final_Pos']:.1f} "  # noqa: 501
                  f"({driver['Position_Change']:.1f} positions)")
    else:
        print("No significant fallers predicted")

    return results


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

    print("Loading F3 qualifying data...")
    f3_qualifying_df = load_qualifying_data('F3')

    f3_df = load_data('F3')

    print("Adding qualifying features...")
    f3_df = calculate_qualifying_features(f3_df, f3_qualifying_df)

    print("Creating target variable for position prediction...")
    f3_df = create_target_variable(f3_df)

    print("Engineering features...")
    features_df = engineer_features(f3_df)
    features_df['target_position'] = f3_df['target_position']

    print("Training position prediction models...")
    models, feature_cols, scaler = train_models(features_df)

    print("Predicting final championship standings...")
    predict_final_championship_standings(models, features_df, feature_cols, scaler)

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


if __name__ == "__main__":
    main()
