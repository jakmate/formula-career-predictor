import os
import joblib
import torch
from datetime import datetime

from app.core.state import AppState
from app.config import MODELS_DIR, LOGGER
from app.core.predictor import train_models, RacingPredictor


class ModelService:
    def __init__(self, app_state: AppState, series: str = None):
        self.app_state = app_state
        self.series = series

    async def save_models(self):
        """Save models to disk"""
        try:
            # Create series-specific directory
            series_dir = os.path.join(MODELS_DIR, self.series) if self.series else MODELS_DIR
            os.makedirs(series_dir, exist_ok=True)

            models_to_save = self.app_state.models[self.series] if self.series else self.app_state.models  # noqa: 501

            for name, model in models_to_save.items():
                if name == "PyTorch":
                    torch.save(
                        model.state_dict(),
                        os.path.join(series_dir, f"{name}.pt"),
                        _use_new_zipfile_serialization=True
                    )
                else:
                    joblib.dump(model, os.path.join(series_dir, f"{name}.joblib"))

            # Save preprocessor
            preprocessor_data = {
                'scaler': self.app_state.scaler[self.series] if self.series else self.app_state.scaler,  # noqa: 501
                'feature_cols': self.app_state.feature_cols[self.series] if self.series else self.app_state.feature_cols  # noqa: 501
            }
            joblib.dump(preprocessor_data, os.path.join(series_dir, "preprocessor.joblib"))

            LOGGER.info(f"Models saved successfully for {self.series or 'all series'}")

        except Exception as e:
            LOGGER.error(f"Error saving models: {e}")

    async def load_models(self):
        """Load models from disk"""
        try:
            models_loaded = False

            # Load for specific series or all series
            series_to_load = [self.series] if self.series else ['f3_to_f2', 'f2_to_f1']

            for series in series_to_load:
                series_dir = os.path.join(MODELS_DIR, series)
                if not os.path.exists(series_dir):
                    continue

                # Load preprocessor
                preprocessor_path = os.path.join(series_dir, "preprocessor.joblib")
                if os.path.exists(preprocessor_path):
                    preprocessor = joblib.load(preprocessor_path)
                    self.app_state.scaler[series] = preprocessor['scaler']
                    self.app_state.feature_cols[series] = preprocessor['feature_cols']

                # Load models
                for model_file in os.listdir(series_dir):
                    if model_file == "preprocessor.joblib":
                        continue

                    name = os.path.splitext(model_file)[0]
                    model_path = os.path.join(series_dir, model_file)

                    if model_file.endswith(".joblib"):
                        model = joblib.load(model_path)
                        self.app_state.models[series][name] = model
                        models_loaded = True
                    elif model_file.endswith(".pt"):
                        model = RacingPredictor(len(self.app_state.feature_cols[series]))
                        state_dict = torch.load(
                            model_path,
                            map_location=torch.device('cpu'),
                            weights_only=False
                        )
                        model.load_state_dict(state_dict)
                        self.app_state.models[series][name] = model
                        models_loaded = True

            if models_loaded:
                # Update available models list
                all_models = []
                for series in ['f3_to_f2', 'f2_to_f1']:
                    all_models.extend([f"{series}_{model}" for model in self.app_state.models[series].keys()])  # noqa: 501
                self.app_state.system_status["models_available"] = all_models
                LOGGER.info(f"Loaded models for series: {list(self.app_state.models.keys())}")

            return models_loaded

        except Exception as e:
            LOGGER.error(f"Error loading models: {e}")
            return False

    async def train_models(self, trainable_df):
        """Train models on provided data"""
        LOGGER.info(f"Training models for {self.series} on {len(trainable_df)} historical records")

        (
            models,
            feature_cols,
            scaler
        ) = train_models(trainable_df)

        # Store in series-specific slots
        self.app_state.models[self.series] = models
        self.app_state.feature_cols[self.series] = feature_cols
        self.app_state.scaler[self.series] = scaler

        self.app_state.system_status["last_training"] = datetime.now()
        self.app_state.system_status["last_trained_season"] = trainable_df['year'].max()

        # Update available models
        all_models = []
        for series in ['f3_to_f2', 'f2_to_f1']:
            if series in self.app_state.models:
                all_models.extend([f"{series}_{model}" for model in self.app_state.models[series].keys()])  # noqa: 501
        self.app_state.system_status["models_available"] = all_models

        self.app_state.system_status["data_health"][self.series] = {
            "historical_records": len(trainable_df),
            "current_records": 0
        }
