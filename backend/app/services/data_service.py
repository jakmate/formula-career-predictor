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

    async def load_current_data(self):
        """Load and process current racing data"""
        f3_df = load_data('F3')
        f2_df = load_standings_data('F2', 'drivers')

        if f3_df.empty:
            raise HTTPException(status_code=404, detail="No F3 data available")

        f3_qualifying_df = load_qualifying_data('F3')
        f3_df = calculate_qualifying_features(f3_df, f3_qualifying_df)
        f3_df = create_target_variable(f3_df, f2_df)
        features_df = engineer_features(f3_df)
        features_df['promoted'] = f3_df['promoted']

        current_df = features_df[features_df['year'] == CURRENT_YEAR].copy()
        if current_df.empty:
            current_year = features_df['year'].max()
            current_df = features_df[features_df['year'] == current_year].copy()

        if current_df.empty:
            raise HTTPException(status_code=404, detail="No drivers data available")

        return current_df

    async def initialize_system(self):
        """Initial data loading and processing"""
        from model_service import ModelService

        LOGGER.info("Initializing system...")

        f3_df = load_data('F3')
        f2_df = load_standings_data('F2', 'drivers')
        f3_qualifying_df = load_qualifying_data('F3')

        f3_df = calculate_qualifying_features(f3_df, f3_qualifying_df)
        f3_df = create_target_variable(f3_df, f2_df)
        features_df = engineer_features(f3_df)
        features_df['promoted'] = f3_df['promoted']

        trainable_df = features_df[features_df['year'] < CURRENT_YEAR]

        if not trainable_df.empty:
            model_service = ModelService(self.app_state)
            await model_service.train_models(trainable_df)
            await model_service.save_models()
            self.app_state.save_state()
        else:
            LOGGER.warning("No historical data available for training")

        # Generate current predictions
        from prediction_service import PredictionService
        prediction_service = PredictionService(self.app_state)
        await prediction_service.update_predictions(features_df)
