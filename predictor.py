import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import glob


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

                    # Clean driver names if Driver column exists
                    if 'Driver' in entries_df.columns:
                        entries_df['Driver'] = entries_df['Driver'].str.strip()
                    if 'Driver' in df.columns:
                        df['Driver'] = df['Driver'].str.strip()

                    # Merge team data if both Driver and Team columns exist
                    if 'Driver' in entries_df.columns and 'Team' in entries_df.columns:  # noqa: E501
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

        # Check next 2 years for F2 participation
        moved = 0
        for offset in [1, 2]:
            target_year = last_f3_year + offset
            if target_year > max_f2_year:
                break

            participation = f2_participation_df[
                (f2_participation_df['driver'] == driver) &
                (f2_participation_df['year'] == target_year)
            ]

            if not participation.empty and participation['participated'].iloc[0]:  # noqa: E501
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


def engineer_features(df):
    """Create features for ML model."""
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

    race_cols = get_race_columns(df)
    wins, podiums, points_finishes, dnfs, races_completed = [], [], [], [], []

    for _, row in df.iterrows():
        driver_wins = 0
        driver_podiums = 0
        driver_points = 0
        driver_dnfs = 0
        driver_races = 0

        for col in race_cols:
            if col not in row or pd.isna(row[col]):
                continue

            result = str(row[col]).strip()
            if not result:
                continue

            # Handle DNF/DNS cases
            if any(x in result for x in ['Ret', 'DNS', 'WD', 'NC', 'DSQ']):
                driver_dnfs += 1
                driver_races += 1
                continue

            try:
                # Extract finishing position
                pos = int(result.split()[0].replace(
                    '†', '').replace('F', '').replace('P', ''))
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
            except:  # noqa: E722
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

    # Calculate rates
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


def train_model(df):
    """Train ML models to predict F2 progression."""
    if df.empty:
        print("No data available for training")
        return {}, None, None, None

    df_clean = df.dropna(subset=['moved_to_f2', 'final_position', 'points'])
    feature_cols = [
        'final_position', 'win_rate', 'podium_rate',
        'dnf_rate', 'points_per_race', 'top_10_rate',
        'years_in_f3'
    ]

    X = df_clean[feature_cols].fillna(0)
    y = df_clean['moved_to_f2']

    print(f"Dataset size: {len(X)} drivers")
    print(f"F2 progressions: {y.sum()} ({y.mean():.2%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(random_state=69,
                                                class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=69,
                                                  class_weight='balanced'),
        'MLP': MLPClassifier(random_state=69)
    }

    results = {}
    for name, model in models.items():
        print(f"\n{name} Results:")
        print("-" * 30)

        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nFeature Importance:")
            print(importance)

        results[name] = model

    return results, X_test, y_test, feature_cols


def predict_current_drivers(models, df, feature_cols):
    """Make predictions for F3 2025 drivers (F2 2026 progression)."""
    if df.empty:
        print("No data available for prediction")
        return

    # Look for F3 2025 data specifically
    f3_2025_drivers = df[df['year'] == 2025].copy()

    if f3_2025_drivers.empty:
        print("No F3 2025 data found for predictions")
        # Fallback to latest year
        current_year = df['year'].max()
        f3_2025_drivers = df[df['year'] == current_year].copy()
        print(f"Using {current_year} F3 data instead")

    if f3_2025_drivers.empty:
        print("No current year data found")
        return

    print("\nPredictions for F3 2025 drivers:")
    print("=" * 70)

    X_current = f3_2025_drivers[feature_cols].fillna(0)

    for name, model in models.items():
        print(f"\n{name} Predictions:")
        print("-" * 50)

        if name != 'Logistic Regression':
            probas = model.predict_proba(X_current)[:, 1]
            predictions = model.predict(X_current)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_current)
            probas = model.predict_proba(X_scaled)[:, 1]
            predictions = model.predict(X_scaled)

        results = pd.DataFrame({
            'Driver': f3_2025_drivers['driver'],
            'Position': f3_2025_drivers['final_position'],
            'Points': f3_2025_drivers['points'],
            'F2_Probability': probas,
            'Prediction': predictions
        }).sort_values('F2_Probability', ascending=False)

        print(results.to_string(index=False, float_format='%.3f'))


print("Loading F3 data...")
f3_df = load_and_combine_data('F3')

print("Loading F2 data...")
f2_df = load_and_combine_data('F2')

if f3_df.empty or f2_df.empty:
    print("No F2/F3 data found. Check file paths.")
    exit()

print("Calculating years in F3 for each driver...")
f3_df = f3_df.sort_values(by=['Driver', 'year'])
f3_df['years_in_f3'] = f3_df.groupby('Driver').cumcount() + 1

print("Creating target variable based on F2 participation...")
f3_df = create_target_using_f2_data(f3_df, f2_df)

print("Engineering features...")
features_df = engineer_features(f3_df)
features_df['moved_to_f2'] = f3_df['moved_to_f2']

print("Training models...")
models, X_test, y_test, feature_cols = train_model(features_df)

print("Making predictions for F3 2025 drivers...")
predict_current_drivers(models, features_df, feature_cols)
