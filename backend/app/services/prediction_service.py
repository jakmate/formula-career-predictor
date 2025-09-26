import time
import torch
from datetime import datetime
from typing import List
from app.models.predictions import PredictionsResponse, ModelResults, PredictionResponse
from app.models.system import SystemStatus
from app.core.state import AppState
from app.services.data_service import DataService
from app.config import LOGGER


class PredictionService:
    def __init__(self, app_state: AppState, series: str, data_service: DataService):
        self.app_state = app_state
        self.series = series
        self.data_service = data_service
        self.prediction_cache = {}

    async def get_predictions(self) -> PredictionsResponse:
        """Get predictions from all models with feature caching"""
        start_time = time.time()

        if not self.app_state.models[self.series]:
            raise ValueError(f"No models available for series {self.series}")

        # Cache key for processed features
        cache_key = f"{self.series}_processed_features"

        data_start = time.time()
        if cache_key not in self.prediction_cache:
            current_df = await self.data_service.load_current_data(self.series)
            feature_cols = self.app_state.feature_cols[self.series]

            if not feature_cols:
                raise ValueError(f"No feature columns available for series {self.series}")

            X_current = current_df[feature_cols].fillna(0)

            # Cache both the processed features and the original dataframe
            self.prediction_cache[cache_key] = {
                'current_df': current_df,
                'X_current': X_current,
                'timestamp': datetime.now()
            }
            LOGGER.info(f"Feature preparation took {time.time() - data_start:.2f}s")
        else:
            cached_data = self.prediction_cache[cache_key]
            current_df = cached_data['current_df']
            X_current = cached_data['X_current']
            LOGGER.info(f"Used cached features in {time.time() - data_start:.2f}s")

        prediction_start = time.time()
        all_predictions = {}
        models = list(self.app_state.models[self.series].keys())

        for model_name in models:
            model_start = time.time()
            try:
                raw_probas = self._get_model_predictions(model_name, X_current)
                predictions = self._create_prediction_responses(current_df, raw_probas)
                all_predictions[model_name] = ModelResults(
                    model_name=model_name,
                    predictions=predictions,
                    accuracy_metrics={"total_predictions": len(predictions)}
                )
                LOGGER.info(f"Model {model_name} prediction took {time.time() - model_start:.2f}s")
            except Exception as e:
                LOGGER.error(f"Error getting predictions for {model_name}: {e}")
                all_predictions[model_name] = ModelResults(
                    model_name=model_name,
                    predictions=[],
                    accuracy_metrics={"total_predictions": 0, "error_count": 1}
                )
        LOGGER.info(f"All model predictions took {time.time() - prediction_start:.2f}s")
        LOGGER.info(f"Total request time: {time.time() - start_time:.2f}s")

        return PredictionsResponse(
            models=models,
            predictions=all_predictions,
            system_status=SystemStatus(**self.app_state.system_status)
        )

    def _get_model_predictions(self, model_name: str, X_current):
        """Extract prediction logic for reusability"""
        if model_name not in self.app_state.models[self.series]:
            raise ValueError(f"Model {model_name} not found for series {self.series}")

        model = self.app_state.models[self.series][model_name]

        if 'PyTorch' in model_name:
            X_current_scaled = self.app_state.scaler[self.series].transform(X_current)
            model.eval()
            with torch.no_grad():
                X_torch = torch.FloatTensor(X_current_scaled)
                if torch.cuda.is_available():
                    X_torch = X_torch.cuda()
                logits = model(X_torch)
                raw_predictions = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            raw_predictions = model.predict_proba(X_current)[:, 1]

        if hasattr(model, 'calibrator') and model.calibrator is not None:
            return model.calibrator.transform(raw_predictions)

        return raw_predictions

    def _create_prediction_responses(
            self, current_df, calibrated_probas) -> List[PredictionResponse]:
        """Create standardized prediction response objects"""
        prediction_values = calibrated_probas * 100.0  # Percentage for classification
        predictions = []

        for idx, (_, row) in enumerate(current_df.iterrows()):
            predictions.append(PredictionResponse(
                driver=row['driver'],
                nationality=row.get('nationality'),
                position=int(row['pos']),
                points=float(row['points']),
                avg_quali_pos=float(row.get('avg_quali_pos', 0)),
                wins=int(row['wins']),
                win_rate=float(row['win_rate']),
                podiums=int(row['podiums']),
                top_10_rate=float(row['top_10_rate']),
                dnf_rate=float(row['dnf_rate']),
                experience=int(row['experience']),
                dob=row.get('dob'),
                age=float(row.get('age', 0)) if row.get('age') else None,
                participation_rate=float(row['participation_rate']),
                teammate_h2h=float(row['teammate_h2h_rate']),
                team=str(row['team']),
                team_pos=int(row['team_pos']),
                team_points=float(row['team_points']),
                nationality_encoded=float(row.get('nationality_encoded', 0)),
                era=int(row.get('era', 0)),
                consistency_score=float(row.get('consistency_score', 0)),
                empirical_percentage=float(prediction_values[idx])
            ))

        predictions.sort(key=lambda x: x.empirical_percentage, reverse=True)
        return predictions

    async def update_predictions(self, features_df=None):
        """Generate predictions for current season"""
        try:
            if features_df is None:
                current_df = await self.data_service.load_current_data(self.series)
            else:
                current_df = features_df[features_df['year'] >= self.app_state.system_status.get('current_year', 2024)].copy() # noqa: 501

            if current_df.empty:
                LOGGER.warning(f"No current data for {self.series} predictions")
                return

            X_current = current_df[self.app_state.feature_cols[self.series]].fillna(0)
            predictions = []
            for model_name in self.app_state.models[self.series].keys():
                try:
                    result = self._get_model_predictions(model_name, X_current)
                    predictions.append({
                        "model": model_name,
                        "series": self.series,
                        "predictions": result,
                        "timestamp": datetime.now()
                    })
                except Exception as e:
                    LOGGER.error(f"Prediction failed for {model_name} in {self.series}: {e}")

            # Store predictions with series key
            if not hasattr(self.app_state, 'current_predictions'):
                self.app_state.current_predictions = {}
            self.app_state.current_predictions[self.series] = predictions
            LOGGER.info(f"Generated {len(predictions)} prediction sets for {self.series}")

        except Exception as e:
            LOGGER.error(f"Prediction update failed for {self.series}: {e}")

    def clear_prediction_cache(self):
        """Clear cached predictions and features"""
        self.prediction_cache.clear()
        LOGGER.info(f"Cleared prediction cache for {self.series}")
