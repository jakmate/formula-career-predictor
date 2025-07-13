import os
import joblib
import torch
from datetime import datetime

from app.core.state import AppState
from app.config import MODELS_DIR, LOGGER
from app.core.predictor import train_models, RacingPredictor


class ModelService:
    def __init__(self, app_state: AppState):
        self.app_state = app_state

    async def save_models(self):
        """Save models to disk"""
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)

            for name, model in self.app_state.models.items():
                if name == "PyTorch":
                    torch.save(
                        model.state_dict(),
                        os.path.join(MODELS_DIR, f"{name}.pt"),
                        _use_new_zipfile_serialization=True
                    )
                else:
                    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))

            joblib.dump({
                'scaler': self.app_state.scaler,
                'feature_cols': self.app_state.feature_cols
            }, os.path.join(MODELS_DIR, "preprocessor.joblib"))

            LOGGER.info("Models saved successfully")

        except Exception as e:
            LOGGER.error(f"Error saving models: {e}")

    async def load_models(self):
        """Load models from disk"""
        try:
            preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
            self.app_state.scaler = preprocessor['scaler']
            self.app_state.feature_cols = preprocessor['feature_cols']

            models_loaded = False
            for model_file in os.listdir(MODELS_DIR):
                name = os.path.splitext(model_file)[0]
                if model_file.endswith(".joblib") and model_file != "preprocessor.joblib":
                    model = joblib.load(os.path.join(MODELS_DIR, model_file))
                    self.app_state.models[name] = model
                    models_loaded = True
                elif model_file.endswith(".pt"):
                    model = RacingPredictor(len(self.app_state.feature_cols))
                    state_dict = torch.load(
                        os.path.join(MODELS_DIR, model_file),
                        map_location=torch.device('cpu'),
                        weights_only=False
                    )
                    model.load_state_dict(state_dict)
                    self.app_state.models[name] = model
                    models_loaded = True

            if models_loaded:
                self.app_state.system_status["models_available"] = list(self.app_state.models.keys())  # noqa: 501
                LOGGER.info(f"Loaded {len(self.app_state.models)} models")

            return models_loaded

        except Exception as e:
            LOGGER.error(f"Error loading models: {e}")
            return False

    async def train_models(self, trainable_df):
        """Train models on provided data"""
        LOGGER.info(f"Training models on {len(trainable_df)} historical records")

        (
            self.app_state.models,
            self.app_state.feature_cols,
            self.app_state.scaler
        ) = train_models(trainable_df)

        self.app_state.system_status["last_training"] = datetime.now()
        self.app_state.system_status["last_trained_season"] = trainable_df['year'].max()
        self.app_state.system_status["models_available"] = list(self.app_state.models.keys())
        self.app_state.system_status["data_health"] = {
            "historical_records": len(trainable_df),
            "current_records": 0
        }
