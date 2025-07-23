from fastapi import HTTPException
from app.core.loader import load_data, load_qualifying_data, load_standings_data
from app.core.state import AppState
from app.config import CURRENT_YEAR, LOGGER
from app.core.predictor import (calculate_qualifying_features,
                                create_target_variable,
                                engineer_features)


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

        feeder_quali_df = load_qualifying_data(feeder_series)
        feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_df)
        feeder_df = create_target_variable(feeder_df, parent_df)
        features_df = engineer_features(feeder_df)
        features_df['promoted'] = feeder_df['promoted']

        current_df = features_df[features_df['year'] == CURRENT_YEAR].copy()
        if current_df.empty:
            current_year = features_df['year'].max()
            current_df = features_df[features_df['year'] == current_year].copy()

        if current_df.empty:
            raise HTTPException(status_code=404, detail="No drivers data available")

        return current_df

    def _parse_series(self, series: str):
        """Parse series string to get feeder and parent series"""
        if series == 'f3_to_f2':
            return 'F3', 'F2'
        elif series == 'f2_to_f1':
            return 'F2', 'F1'
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

                feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_df)
                feeder_df = create_target_variable(feeder_df, parent_df)
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
                prediction_service = PredictionService(self.app_state)
                await prediction_service.update_predictions(features_df)
            except Exception as e:
                LOGGER.error(f"Failed to initialize {series}: {e}")

        self.app_state.save_state()
