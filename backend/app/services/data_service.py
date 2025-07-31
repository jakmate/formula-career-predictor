from fastapi import HTTPException
from app.core.loader import load_data, load_qualifying_data, load_standings_data
from app.core.state import AppState
from app.config import CURRENT_YEAR, LOGGER


class DataService:
    def __init__(self, app_state: AppState):
        self.app_state = app_state

    async def load_current_data(self, series: str):
        """Load and process current racing data"""
        # Parse series to get feeder and parent series
        feeder_series, parent_series = self._parse_series(series)

        feeder_df = load_data(feeder_series)
        parent_df = load_standings_data(parent_series, 'drivers')

        if feeder_df.empty:
            raise HTTPException(status_code=404, detail=f"No {feeder_series} data available")

        from app.core.predictor import (calculate_qualifying_features,
                                        create_target_variable,
                                        engineer_features)

        feeder_quali_df = load_qualifying_data(feeder_series)
        feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_df)
        feeder_df = create_target_variable(feeder_df, parent_df, parent_series)
        features_df = engineer_features(feeder_df)
        features_df['promoted'] = feeder_df['promoted']

        current_df = features_df[features_df['year'] == CURRENT_YEAR].copy()
        if current_df.empty:
            current_year = features_df['year'].max()
            current_df = features_df[features_df['year'] == current_year].copy()

        if current_df.empty:
            raise HTTPException(status_code=404, detail="No drivers data available")

        return current_df

    async def load_regression_data(self, series: str):
        """Load data for position regression"""
        series_name = series.replace('_regression', '').upper()

        df = load_data(series_name)
        quali_df = load_qualifying_data(series_name)

        from app.core.regressor import (calculate_qualifying_features,
                                        create_target_variable,
                                        engineer_features)

        df = calculate_qualifying_features(df, quali_df)
        df = create_target_variable(df)  # This should create target_position
        features_df = engineer_features(df)
        features_df['target_position'] = df['target_position']

        current_df = features_df[features_df['year'] == CURRENT_YEAR].copy()
        if current_df.empty:
            current_year = features_df['year'].max()
            current_df = features_df[features_df['year'] == current_year].copy()

        return current_df

    def _parse_series(self, series: str):
        """Parse series string to get feeder and parent series"""
        if series == 'f3_to_f2':
            return 'F3', 'F2'
        elif series == 'f2_to_f1':
            return 'F2', 'F1'
        elif series == 'f1_regression':
            return 'F1'
        elif series == 'f2_regression':
            return 'F2'
        elif series == 'f3_regression':
            return 'F3'
        else:
            raise ValueError(f"Unknown series: {series}")

    async def initialize_system(self):
        """Initial data loading and processing"""
        from app.services.model_service import ModelService

        for series in ['f3_to_f2', 'f2_to_f1']:
            try:
                LOGGER.info(f"Initializing system for {series}...")

                feeder_series, parent_series = self._parse_series(series)

                feeder_df = load_data(feeder_series)
                parent_df = load_standings_data(parent_series, 'drivers')
                feeder_quali_df = load_qualifying_data(feeder_series)

                from app.core.predictor import (calculate_qualifying_features,
                                                create_target_variable,
                                                engineer_features)

                feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_df)
                feeder_df = create_target_variable(feeder_df, parent_df, parent_series)
                features_df = engineer_features(feeder_df)
                features_df['promoted'] = feeder_df['promoted']

                trainable_df = features_df[features_df['year'] < CURRENT_YEAR]

                if not trainable_df.empty:
                    model_service = ModelService(self.app_state, series)
                    await model_service.train_models(trainable_df)
                    await model_service.save_models()
                else:
                    LOGGER.warning(f"No historical data available for training {series}")

                # Generate current predictions
                from app.services.prediction_service import PredictionService
                prediction_service = PredictionService(self.app_state, series)
                await prediction_service.update_predictions(features_df)
            except Exception as e:
                LOGGER.error(f"Failed to initialize {series}: {e}")

        for series in ['f3_regression', 'f2_regression', 'f1_regression']:
            try:
                LOGGER.info(f"Initializing system for {series}...")

                feeder_series = self._parse_series(series)

                feeder_df = load_data(feeder_series)
                feeder_quali_df = load_qualifying_data(feeder_series)

                from app.core.regressor import (calculate_qualifying_features,
                                                create_target_variable,
                                                engineer_features)

                feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_df)
                feeder_df = create_target_variable(feeder_df)
                features_df = engineer_features(feeder_df)
                features_df['target_position'] = feeder_df['target_position']

                trainable_df = features_df[features_df['year'] < CURRENT_YEAR]

                if not trainable_df.empty:
                    model_service = ModelService(self.app_state, series)
                    await model_service.train_models(trainable_df)
                    await model_service.save_models()
                else:
                    LOGGER.warning(f"No historical data available for training {series}")

                # Generate current predictions
                from app.services.prediction_service import PredictionService
                prediction_service = PredictionService(self.app_state, series)
                await prediction_service.update_predictions(features_df)
            except Exception as e:
                LOGGER.error(f"Failed to initialize {series}: {e}")

        self.app_state.save_state()
