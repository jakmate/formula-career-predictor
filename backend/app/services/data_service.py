import time
from fastapi import HTTPException
from app.core.loader import load_data, load_qualifying_data, load_standings_data
from app.core.state import AppState
from app.config import CURRENT_YEAR, LOGGER


class DataService:
    def __init__(self, app_state: AppState, data_cache: dict = None):
        self.app_state = app_state
        self.data_cache = data_cache if data_cache is not None else {}

    async def load_current_data(self, series: str):
        """Load and process current racing data with caching"""
        start_time = time.time()
        cache_key = f"current_data_{series}"

        # Return cached data if available
        if cache_key in self.data_cache:
            LOGGER.info(f"Cache HIT for {series} - returned in {time.time() - start_time:.2f}s")
            return self.data_cache[cache_key]

        LOGGER.info(f"Cache MISS for {series} - processing data...")
        load_start = time.time()

        # Parse series to get feeder and parent series
        feeder_series, parent_series = self._parse_series(series)

        feeder_df = load_data(feeder_series)
        parent_df = load_standings_data(parent_series, 'drivers')
        LOGGER.info(f"Data loading took {time.time() - load_start:.2f}s")

        if feeder_df.empty:
            raise HTTPException(status_code=404, detail=f"No {feeder_series} data available")

        processing_start = time.time()
        from app.core.predictor import (calculate_qualifying_features,
                                        create_target_variable,
                                        engineer_features)

        feeder_quali_df = load_qualifying_data(feeder_series)
        feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_df)
        feeder_df = create_target_variable(feeder_df, parent_df, parent_series)
        features_df = engineer_features(feeder_df)
        features_df['promoted'] = feeder_df['promoted']
        LOGGER.info(f"Feature processing took {time.time() - processing_start:.2f}s")

        current_df = features_df[features_df['year'] == CURRENT_YEAR].copy()
        if current_df.empty:
            current_year = features_df['year'].max()
            current_df = features_df[features_df['year'] == current_year].copy()

        if current_df.empty:
            raise HTTPException(status_code=404, detail="No drivers data available")

        # Cache the processed data
        self.data_cache[cache_key] = current_df
        LOGGER.info(f"Total processing for {series}: {time.time() - start_time:.2f}s - cached {len(current_df)} records") # noqa: 501

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

                from app.core.predictor import (calculate_qualifying_features,
                                                create_target_variable,
                                                engineer_features)

                feeder_df = calculate_qualifying_features(feeder_df, feeder_quali_df)
                feeder_df = create_target_variable(feeder_df, parent_df, parent_series)
                features_df = engineer_features(feeder_df)
                features_df['promoted'] = feeder_df['promoted']

                # Cache the full processed data during initialization
                cache_key = f"full_data_{series}"
                self.data_cache[cache_key] = features_df

                trainable_df = features_df[features_df['year'] < CURRENT_YEAR]

                if not trainable_df.empty:
                    model_service = ModelService(self.app_state, series)
                    await model_service.train_models(trainable_df)
                    await model_service.save_models()
                else:
                    LOGGER.warning(f"No historical data available for training {series}")

                # Generate current predictions
                from app.services.prediction_service import PredictionService
                prediction_service = PredictionService(self.app_state, series, self)
                await prediction_service.update_predictions(features_df)
            except Exception as e:
                LOGGER.error(f"Failed to initialize {series}: {e}")

        self.app_state.save_state()

    def clear_cache(self, series: str = None):
        """Clear cached data for specific series or all"""
        if series:
            keys_to_remove = [k for k in self.data_cache.keys() if series in k]
            for key in keys_to_remove:
                del self.data_cache[key]
            LOGGER.info(f"Cleared cache for {series}")
        else:
            self.data_cache.clear()
            LOGGER.info("Cleared all cached data")
