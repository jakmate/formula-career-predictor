import asyncio
import logging
import pandas as pd
import torch
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from scraper import scrape
from predictor import (
    load_and_combine_data,
    load_team_standings,
    load_qualifying_data,
    enhance_with_team_data,
    calculate_qualifying_features,
    create_target_variable,
    engineer_features,
    train_models,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class PredictionResponse(BaseModel):
    driver: str
    nationality: Optional[str] = None
    position: int
    avg_finish_pos: float
    std_finish_pos: float
    avg_quali_pos: Optional[float] = None
    std_quali_pos: Optional[float] = None
    points: float
    win_rate: float
    podium_rate: float
    top_10_rate: float
    dnf_rate: float
    experience: int
    age: Optional[float]
    has_academy: int
    avg_pos_diff: float
    teammate_battles: float
    team_pos: int
    team_points: float
    points_vs_team_strength: float
    pos_vs_team_strength: float
    raw_probability: float
    empirical_percentage: float
    prediction: int


class ModelResults(BaseModel):
    model_name: str
    predictions: List[PredictionResponse]
    accuracy_metrics: Dict[str, float]


class SystemStatus(BaseModel):
    last_scrape: Optional[datetime]
    last_training: Optional[datetime]
    models_available: List[str]
    data_health: Dict[str, int]


# Global State
class AppState:
    def __init__(self):
        self.ml_models = {}
        self.deep_models = {}
        self.feature_cols = []
        self.scaler = None
        self.system_status = {
            "last_scrape": None,
            "last_training": None,
            "models_available": [],
            "data_health": {}
        }
        self.scheduler = AsyncIOScheduler()


app_state = AppState()


async def scrape_data_task():
    """Background task to scrape racing data"""
    try:
        logger.info("Starting data scraping task...")
        await asyncio.get_event_loop().run_in_executor(None, scrape)
        app_state.system_status["last_scrape"] = datetime.now()
        logger.info("Data scraping completed successfully")
    except Exception as e:
        logger.error(f"Error during data scraping: {e}")


async def train_models_task():
    """Background task to train ML models"""
    try:
        logger.info("Starting model training task...")

        # Load and process data
        f3_df = load_and_combine_data('F3')
        f2_df = load_and_combine_data('F2')

        if f3_df.empty or f2_df.empty:
            logger.warning("No data available for training")
            return

        f3_team_df = load_team_standings('F3')
        f3_df = enhance_with_team_data(f3_df, f3_team_df)
        f3_qualifying_df = load_qualifying_data('F3')
        f3_df = calculate_qualifying_features(f3_df, f3_qualifying_df)
        f3_df = create_target_variable(f3_df, f2_df)
        features_df = engineer_features(f3_df)
        features_df['moved_to_f2'] = f3_df['moved_to_f2']

        # Train models using the unified function
        (app_state.ml_models,
         app_state.deep_models,
         _, _,
         app_state.feature_cols,
         app_state.scaler) = train_models(features_df)

        # Update system status
        app_state.system_status["last_training"] = datetime.now()
        app_state.system_status["models_available"] = (
            list(app_state.ml_models.keys()) + list(app_state.deep_models.keys())
        )
        app_state.system_status["data_health"] = {
            "f3_records": len(f3_df),
            "f2_records": len(f2_df),
            "features_records": len(features_df)
        }

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Error during model training: {e}")


def _get_model_predictions(model_name: str, X_current):
    """Extract prediction logic for reusability"""
    if model_name in app_state.ml_models:
        model = app_state.ml_models[model_name]
        raw_probas = model.predict_proba(X_current)[:, 1]
    elif model_name in app_state.deep_models:
        model = app_state.deep_models[model_name]
        X_current_scaled = app_state.scaler.transform(X_current)

        if 'PyTorch' in model_name:
            model.eval()
            with torch.no_grad():
                X_torch = torch.FloatTensor(X_current_scaled)
                if torch.cuda.is_available():
                    X_torch = X_torch.cuda()
                logits = model(X_torch)
                raw_probas = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            raw_probas = model.predict(X_current_scaled, verbose=0).flatten()
    else:
        raise ValueError(f"Model {model_name} not found")

    return raw_probas


def _create_prediction_responses(current_df, raw_probas):
    """Create standardized prediction response objects"""
    # Apply calibration if available
    model = (app_state.ml_models.get(list(app_state.ml_models.keys())[0]) or
             app_state.deep_models.get(list(app_state.deep_models.keys())[0]))

    if hasattr(model, 'calibrator'):
        empirical_pct = model.calibrator.transform(raw_probas) * 100.0
    else:
        empirical_pct = raw_probas * 100

    predictions_binary = (empirical_pct >= 50).astype(int)

    predictions = []
    for idx, (_, row) in enumerate(current_df.iterrows()):
        predictions.append(PredictionResponse(
            driver=row['driver'],
            nationality=row['nationality'] if pd.notna(row['nationality']) else None,
            position=int(row['final_pos']),
            points=float(row['points']),
            avg_finish_pos=float(row['avg_finish_pos']),
            std_finish_pos=float(row['std_finish_pos']),
            avg_quali_pos=float(row['avg_quali_pos']) if pd.notna(row['avg_quali_pos']) else None,
            std_quali_pos=float(row['std_quali_pos']) if pd.notna(row['std_quali_pos']) else None,
            win_rate=float(row['win_rate']),
            podium_rate=float(row['podium_rate']),
            top_10_rate=float(row['top_10_rate']),
            dnf_rate=float(row['dnf_rate']),
            experience=int(row['years_in_f3']),
            age=float(row['age']) if pd.notna(row['age']) else None,
            has_academy=int(row['has_academy']),
            avg_pos_diff=float(row['avg_pos_vs_teammates']),
            teammate_battles=float(row['teammate_battles']),
            team_pos=int(row['team_pos']),
            team_points=float(row['team_points']),
            points_vs_team_strength=float(row['points_vs_team_strength']),
            pos_vs_team_strength=float(row['pos_vs_team_strength']),
            raw_probability=float(raw_probas[idx]),
            empirical_percentage=float(empirical_pct[idx]),
            prediction=int(predictions_binary[idx])
        ))

    # Sort by empirical percentage (highest first)
    predictions.sort(key=lambda x: x.empirical_percentage, reverse=True)
    return predictions


def _load_current_data():
    """Load and process current racing data"""
    f3_df = load_and_combine_data('F3')
    f2_df = load_and_combine_data('F2')

    if f3_df.empty:
        raise HTTPException(status_code=404, detail="No F3 data available")

    f3_team_df = load_team_standings('F3')
    f3_df = enhance_with_team_data(f3_df, f3_team_df)
    f3_qualifying_df = load_qualifying_data('F3')
    f3_df = calculate_qualifying_features(f3_df, f3_qualifying_df)
    f3_df = create_target_variable(f3_df, f2_df)

    features_df = engineer_features(f3_df)
    features_df['moved_to_f2'] = f3_df['moved_to_f2']

    # Get current year data
    current_year = 2025
    current_df = features_df[features_df['year'] == current_year].copy()
    if current_df.empty:
        current_year = features_df['year'].max()
        current_df = features_df[features_df['year'] == current_year].copy()

    if current_df.empty:
        raise HTTPException(status_code=404, detail="No current drivers data available")

    return current_df


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_task = None

    try:
        logger.info("Starting F3/F2 Predictions API...")

        # Schedule tasks
        app_state.scheduler.add_job(
            scrape_data_task,
            CronTrigger(day_of_week='fri', hour=9, minute=0),
            id='scrape_data',
            replace_existing=True
        )

        app_state.scheduler.add_job(
            train_models_task,
            CronTrigger(day_of_week='fri', hour=10, minute=0),
            id='train_models',
            replace_existing=True
        )

        if not app_state.scheduler.running:
            app_state.scheduler.start()

        # Load existing models in background
        startup_task = asyncio.create_task(train_models_task())

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    try:
        logger.info("Starting application shutdown...")

        # Cancel startup task if still running
        if startup_task and not startup_task.done():
            startup_task.cancel()
            try:
                await startup_task
            except asyncio.CancelledError:
                pass

        if app_state.scheduler.running:
            app_state.scheduler.shutdown(wait=False)

        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


app = FastAPI(
    title="F3/F2 Racing Predictions API",
    version="1.0.0",
    description="API for predicting F3 to F2 driver transitions",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {"message": "F3/F2 Racing Predictions API is running"}


@app.get("/status", response_model=SystemStatus, tags=["System"])
async def get_system_status():
    """Get system status information"""
    return SystemStatus(**app_state.system_status)


@app.post("/scrape", tags=["Data"])
async def trigger_scrape(background_tasks: BackgroundTasks):
    """Manually trigger data scraping"""
    background_tasks.add_task(scrape_data_task)
    return {"message": "Data scraping started in background"}


@app.post("/train", tags=["Models"])
async def trigger_training(background_tasks: BackgroundTasks):
    """Manually trigger model training"""
    background_tasks.add_task(train_models_task)
    return {"message": "Model training started in background"}


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models"""
    return {
        "ml_models": list(app_state.ml_models.keys()),
        "deep_learning_models": list(app_state.deep_models.keys()),
        "total_models": len(app_state.ml_models) + len(app_state.deep_models)
    }


@app.get("/predictions/{model_name}", response_model=ModelResults, tags=["Predictions"])
async def get_predictions(model_name: str) -> ModelResults:
    """Get predictions from a specific model"""
    if model_name not in app_state.ml_models and model_name not in app_state.deep_models:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Load and process data
        current_df = _load_current_data()
        X_current = current_df[app_state.feature_cols].fillna(0)

        # Generate predictions
        raw_probas = _get_model_predictions(model_name, X_current)
        predictions = _create_prediction_responses(current_df, raw_probas)

        return ModelResults(
            model_name=model_name,
            predictions=predictions,
            accuracy_metrics={"total_predictions": len(predictions)}
        )

    except Exception as e:
        logger.error(f"Error generating predictions for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions", tags=["Predictions"])
async def get_all_predictions():
    """Get predictions from all available models"""
    all_predictions = {}

    for model_name in list(app_state.ml_models.keys()) + list(app_state.deep_models.keys()):
        try:
            result = await get_predictions(model_name)
            all_predictions[model_name] = result
        except Exception as e:
            logger.error(f"Error getting predictions for {model_name}: {e}")
            all_predictions[model_name] = {"error": str(e)}

    return all_predictions

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
