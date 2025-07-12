import asyncio
import json
import logging
import os
import joblib
import pandas as pd
import pytz
import torch
# import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from loader import load_data, load_qualifying_data, load_standings_data
from scraping.scrape import scrape_current_year
from predictor import (
    RacingPredictor,
    calculate_qualifying_features,
    create_target_variable,
    engineer_features,
    train_models,
)

# Configuration
MODELS_DIR = "models"
STATE_FILE = "system_state.json"
CURRENT_YEAR = datetime.now().year
SEASON_END_MONTH = 12
SCHEDULE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'schedules', str(CURRENT_YEAR))
PORT = int(os.environ.get("PORT", 8000))

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
    wins: int
    podiums: int
    win_rate: float
    podium_rate: float
    top_10_rate: float
    dnf_rate: float
    experience: int
    age: Optional[float]
    dob: Optional[date] = None
    participation_rate: float
    pole_rate: float
    top_10_starts_rate: float
    teammate_h2h: float
    team: str
    team_pos: int
    team_points: float
    points_share: float
    raw_probability: float
    empirical_percentage: float


class ModelResults(BaseModel):
    model_name: str
    predictions: List[PredictionResponse]
    accuracy_metrics: Dict[str, float]


class SystemStatus(BaseModel):
    last_scrape: Optional[datetime]
    last_training: Optional[datetime]
    models_available: List[str]
    data_health: Dict[str, int]


class RefreshResponse(BaseModel):
    message: str
    task_id: Optional[str] = None
    estimated_completion: Optional[datetime] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: int
    last_training: Optional[datetime]


class AllPredictionsResponse(BaseModel):
    models: List[str]
    predictions: Dict[str, ModelResults]
    system_status: SystemStatus


# Global State
class AppState:
    def __init__(self):
        self.models = {}
        self.feature_cols = []
        self.scaler = None
        self.current_predictions = []
        self.system_status = {
            "last_scrape": None,
            "last_training": None,
            "last_trained_season": None,
            "models_available": [],
            "data_health": {}
        }
        self.scheduler = AsyncIOScheduler()

    def save_state(self):
        """Save critical state to disk"""
        state = {
            "last_scrape": self.system_status["last_scrape"].isoformat()
            if self.system_status["last_scrape"] else None,
            "last_training": self.system_status["last_training"].isoformat()
            if self.system_status["last_training"] else None,
            "last_trained_season": self.system_status["last_trained_season"],
            "models_available": self.system_status["models_available"],
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, default=str)

    def load_state(self):
        """Load state from disk"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)

                self.system_status["last_scrape"] = (
                    datetime.fromisoformat(state["last_scrape"])
                    if state["last_scrape"] else None
                )
                self.system_status["last_training"] = (
                    datetime.fromisoformat(state["last_training"])
                    if state["last_training"] else None
                )
                self.system_status["last_trained_season"] = state["last_trained_season"]
                self.system_status["models_available"] = state["models_available"]

                return True

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted state file: {e}. Reinitializing state.")
            # Backup corrupted file
            os.rename(STATE_FILE, f"{STATE_FILE}.backup")
            return False

        except Exception as e:
            logger.error(f"Error loading state: {e}")

        return False


app_state = AppState()


# System Initialization
def initialize_system():
    """Initial data loading and processing"""
    logger.info("Initializing system...")

    # Load data
    f3_df = load_data('F3')
    f2_df = load_standings_data('F2', 'drivers')
    f3_qualifying_df = load_qualifying_data('F3')

    # Process data
    f3_df = calculate_qualifying_features(f3_df, f3_qualifying_df)
    f3_df = create_target_variable(f3_df, f2_df)
    features_df = engineer_features(f3_df)
    features_df['promoted'] = f3_df['promoted']

    # Train models on all available historical data
    trainable_df = features_df[
        features_df['year'] < CURRENT_YEAR  # Only seasons with known outcomes
    ]

    if not trainable_df.empty:
        logger.info(f"Training models on {len(trainable_df)} historical records")
        (
            app_state.models,
            _, _,
            app_state.feature_cols,
            app_state.scaler,
            _, _
        ) = train_models(trainable_df)

        # Update system status
        app_state.system_status["last_training"] = datetime.now()
        app_state.system_status["last_trained_season"] = trainable_df['year'].max()
        app_state.system_status["models_available"] = (list(app_state.models.keys()))
        app_state.system_status["data_health"] = {
            "historical_records": len(trainable_df),
            "current_records": len(features_df[features_df['year'] >= CURRENT_YEAR])
        }

        # Save models and state
        save_models()
        app_state.save_state()
    else:
        logger.warning("No historical data available for training")

    # Generate current predictions
    update_predictions(features_df)


def save_models():
    """Save models to disk"""
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Save models
        for name, model in app_state.models.items():
            if name == "PyTorch":
                torch.save(
                    model.state_dict(),
                    os.path.join(MODELS_DIR, f"{name}.pt"),
                    _use_new_zipfile_serialization=True
                )
            else:
                joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))

        # Save scaler and features
        joblib.dump({
            'scaler': app_state.scaler,
            'feature_cols': app_state.feature_cols
        }, os.path.join(MODELS_DIR, "preprocessor.joblib"))

        logger.info("Models saved successfully")

    except Exception as e:
        logger.error(f"Error saving models: {e}")


def load_models():
    """Load models from disk"""
    try:
        # Load preprocessor
        try:
            preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
            app_state.scaler = preprocessor['scaler']
            app_state.feature_cols = preprocessor['feature_cols']
            logger.info("Preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            return False

        models_loaded = False

        # Load models
        for model_file in os.listdir(MODELS_DIR):
            name = os.path.splitext(model_file)[0]
            if model_file.endswith(".joblib") and model_file != "preprocessor.joblib":
                try:
                    model = joblib.load(os.path.join(MODELS_DIR, model_file))
                    app_state.models[name] = model
                    logger.info(f"Loaded model: {name}")
                    models_loaded = True
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {e}")
            elif model_file.endswith(".pt"):
                try:
                    model = RacingPredictor(len(app_state.feature_cols))

                    # Load with explicit device mapping
                    state_dict = torch.load(
                        os.path.join(MODELS_DIR, model_file),
                        map_location=torch.device('cpu'),
                        weights_only=False
                    )
                    model.load_state_dict(state_dict)
                    app_state.models[name] = model
                    logger.info(f"Loaded model: {name}")
                    models_loaded = True
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {e}")

        # Update status
        if models_loaded:
            app_state.system_status["models_available"] = (list(app_state.models.keys()))
            logger.info(f"Loaded {len(app_state.models)} models")
            return True

        logger.error("No models loaded successfully")
        return False

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False


def update_predictions(features_df=None):
    """Generate predictions for current season"""
    try:
        if features_df is None:
            # Reload current data
            current_df = _load_current_data()
        else:
            current_df = features_df[features_df['year'] >= CURRENT_YEAR].copy()

        if current_df.empty:
            logger.warning("No current data for predictions")
            return

        # Extract only the feature columns for model prediction
        X_current = current_df[app_state.feature_cols].fillna(0)

        # Generate predictions
        predictions = []
        for model_name in app_state.system_status["models_available"]:
            try:
                result = _get_model_predictions(model_name, X_current)
                predictions.append({
                    "model": model_name,
                    "predictions": result,
                    "timestamp": datetime.now()
                })
            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {e}")

        app_state.current_predictions = predictions
        logger.info(f"Generated {len(predictions)} prediction sets")

    except Exception as e:
        logger.error(f"Prediction update failed: {e}")


def is_season_complete():
    """Check if current season is complete based on date"""
    now = datetime.now()
    return now.month > SEASON_END_MONTH and now.year == CURRENT_YEAR


# Background tasks
async def scrape_and_train_task():
    """Combined scraping and training task for new seasons"""
    try:
        logger.info("Starting data scraping task...")
        await asyncio.get_event_loop().run_in_executor(None, scrape_current_year)
        app_state.system_status["last_scrape"] = datetime.now()
        logger.info("Data scraping completed")

        # Check if we have a complete new season to train on
        if is_season_complete() and CURRENT_YEAR > app_state.system_status["last_trained_season"]:
            logger.info(f"New season {CURRENT_YEAR} complete. Starting training...")
            await train_models_task()
        else:
            logger.info("No new complete season available. Updating predictions only.")
            update_predictions()
    except Exception as e:
        logger.error(f"Scrape and train task failed: {e}")
    finally:
        app_state.save_state()


async def train_models_task():
    """Train models on newly available complete seasons"""
    try:
        # Load newly scraped data
        f3_df = load_data('F3')
        f2_df = load_standings_data('F2', 'drivers')
        f3_qualifying_df = load_qualifying_data('F3')

        # Process data
        f3_df = calculate_qualifying_features(f3_df, f3_qualifying_df)
        f3_df = create_target_variable(f3_df, f2_df)
        features_df = engineer_features(f3_df)
        features_df['promoted'] = f3_df['promoted']

        # Train only on newly complete seasons
        trainable_df = features_df[
            (features_df['year'] == CURRENT_YEAR) &
            (features_df['year'] > app_state.system_status["last_trained_season"])
        ]

        if trainable_df.empty:
            logger.warning("No new trainable data available")
            return

        logger.info(f"Training models on {len(trainable_df)} new records")
        (
            app_state.models,
            _, _,
            app_state.feature_cols,
            app_state.scaler,
            _, _
        ) = train_models(trainable_df)

        # Update system status
        app_state.system_status["last_training"] = datetime.now()
        app_state.system_status["last_trained_season"] = CURRENT_YEAR
        app_state.system_status["models_available"] = (list(app_state.models.keys()))
        app_state.system_status["data_health"]["historical_records"] = len(trainable_df)

        # Save models and update predictions
        save_models()
        update_predictions(features_df)
        app_state.save_state()
    except Exception as e:
        logger.error(f"Training task failed: {e}")


# Helper functions
def _get_model_predictions(model_name: str, X_current):
    """Extract prediction logic for reusability"""
    if model_name in app_state.models:
        model = app_state.models[model_name]
        if 'PyTorch' in model_name:
            X_current_scaled = app_state.scaler.transform(X_current)
            model.eval()
            with torch.no_grad():
                X_torch = torch.FloatTensor(X_current_scaled)
                if torch.cuda.is_available():
                    X_torch = X_torch.cuda()
                logits = model(X_torch)
                raw_probas = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            raw_probas = model.predict_proba(X_current)[:, 1]

        if hasattr(model, 'calibrator') and model.calibrator is not None:
            calibrated_probas = model.calibrator.transform(raw_probas)
            return calibrated_probas
    else:
        raise ValueError(f"Model {model_name} not found")

    return raw_probas


def _create_prediction_responses(current_df, calibrated_probas):
    """Create standardized prediction response objects"""
    # Apply calibration if available
    empirical_pct = calibrated_probas * 100.0

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
            wins=int(row['wins']),
            win_rate=float(row['win_rate']),
            podiums=int(row['podiums']),
            podium_rate=float(row['podium_rate']),
            top_10_rate=float(row['top_10_rate']),
            dnf_rate=float(row['dnf_rate']),
            experience=int(row['experience']),
            dob=row['dob'],
            age=float(row['age']) if pd.notna(row['age']) else None,
            participation_rate=float(row['participation_rate']),
            teammate_h2h=float(row['teammate_h2h_rate']),
            pole_rate=float(row['pole_rate']),
            top_10_starts_rate=float(row['top_10_starts_rate']),
            team=str(row['team']),
            team_pos=int(row['team_pos']),
            team_points=float(row['team_points']),
            points_share=float(row['points_share']),
            raw_probability=float(calibrated_probas[idx]),
            empirical_percentage=float(empirical_pct[idx]),
        ))

    # Sort by empirical percentage (highest first)
    predictions.sort(key=lambda x: x.empirical_percentage, reverse=True)
    return predictions


def _load_current_data():
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

    # Get current year data
    current_year = datetime.now().year
    current_df = features_df[features_df['year'] == current_year].copy()
    if current_df.empty:
        current_year = features_df['year'].max()
        current_df = features_df[features_df['year'] == current_year].copy()

    if current_df.empty:
        raise HTTPException(status_code=404, detail="No current drivers data available")

    return current_df


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown handling"""
    try:
        logger.info("Starting application...")

        # Initialize system state
        app_state.load_state()

        # Try to load pre-trained models
        if not load_models():
            logger.info("No models found. Initializing system...")
            initialize_system()

        # Set up scheduled tasks
        app_state.scheduler.add_job(
            scrape_and_train_task,
            'cron', day_of_week='mon', hour=3,  # Every Monday at 3 AM
            id='weekly_scrape_train'
        )
        app_state.scheduler.start()
        logger.info("Scheduler started")

        yield
    finally:
        logger.info("Shutting down application...")
        app_state.scheduler.shutdown()
        app_state.save_state()
        logger.info("Application shutdown complete")


app = FastAPI(
    title="F3/F2 Career Predictions API",
    version="1.0.0",
    description="API for predicting F3 to F2 driver transitions",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://formula-predictions-frontend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "name": "F3/F2 Racing Predictions API",
        "status": "running",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check with system status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": len(app_state.models),
        "last_training": app_state.system_status.get("last_training")
    }


@app.get("/api/predictions", response_model=AllPredictionsResponse, tags=["Predictions"])
async def get_all_predictions():
    """Get predictions from all models"""
    try:
        # Load and process data
        current_df = _load_current_data()
        X_current = current_df[app_state.feature_cols].fillna(0)

        all_predictions = {}
        models = list(app_state.models.keys())

        for model_name in models:
            try:
                raw_probas = _get_model_predictions(model_name, X_current)
                predictions = _create_prediction_responses(current_df, raw_probas)

                all_predictions[model_name] = ModelResults(
                    model_name=model_name,
                    predictions=predictions,
                    accuracy_metrics={"total_predictions": len(predictions)}
                )
            except Exception as e:
                logger.error(f"Error getting predictions for {model_name}: {e}")
                all_predictions[model_name] = ModelResults(
                    model_name=model_name,
                    predictions=[],
                    accuracy_metrics={"error": str(e)}
                )

        return AllPredictionsResponse(
            models=models,
            predictions=all_predictions,
            system_status=SystemStatus(**app_state.system_status)
        )

    except Exception as e:
        logger.error(f"Error in get_all_predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh", response_model=RefreshResponse, tags=["System"])
async def refresh_data(background_tasks: BackgroundTasks):
    """Trigger data refresh and model retraining"""
    background_tasks.add_task(scrape_and_train_task)
    return RefreshResponse(
        message="Data refresh and training started in background",
        estimated_completion=datetime.now() + timedelta(minutes=2)
    )


@app.get("/api/races/{series}", tags=["Schedule"])
async def get_series_schedule(
    series: str,
    timezone: Optional[str] = None,
    x_timezone: Optional[str] = Header(None)
):
    """Get schedule for a specific racing series with timezone conversion"""
    valid_series = ['f1', 'f2', 'f3']
    if series not in valid_series:
        raise HTTPException(status_code=404, detail="Invalid series specified")

    file_path = os.path.join(SCHEDULE_DIR, f"{series}.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Schedule data not found")

    try:
        with open(file_path, 'r') as f:
            schedule = json.load(f)

        # Get timezone from query param, header, or default to UTC
        user_timezone = timezone or x_timezone or 'UTC'

        # Convert times if timezone is provided and not UTC
        if user_timezone != 'UTC':
            schedule = convert_schedule_timezone(schedule, user_timezone)

        return schedule
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading schedule: {str(e)}")


@app.get("/api/races/{series}/next", tags=["Schedule"])
async def get_next_race(
    series: str,
    timezone: Optional[str] = None,
    x_timezone: Optional[str] = Header(None)
):
    """Get the next upcoming race for a series with timezone conversion"""
    valid_series = ['f1', 'f2', 'f3']
    if series not in valid_series:
        raise HTTPException(status_code=404, detail="Invalid series specified")

    file_path = os.path.join(SCHEDULE_DIR, f"{series}.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Schedule data not found")

    try:
        with open(file_path, 'r') as f:
            schedule = json.load(f)

        total_rounds = len(schedule)
        now = datetime.utcnow()
        next_race = None
        next_session = None

        for race in schedule:
            for session_name, session_info in race['sessions'].items():
                if session_info.get('time') == 'TBC':
                    continue

                start_str = session_info.get('start')
                if not start_str:
                    continue

                try:
                    session_dt = datetime.fromisoformat(start_str)
                except Exception as e:
                    print(e)
                    continue

                if session_dt > now:
                    candidate_session = {
                        'name': session_name,
                        'date': start_str
                    }

                    if not next_session or session_dt < datetime.fromisoformat(next_session['date']):  # noqa: 501
                        next_session = candidate_session
                        if not next_race:
                            next_race = race
                            next_race['totalRounds'] = total_rounds

        if next_race and next_session:
            next_race['nextSession'] = next_session

            # Convert timezone if provided
            user_timezone = timezone or x_timezone or 'UTC'
            if user_timezone != 'UTC':
                next_race = convert_race_timezone(next_race, user_timezone)

        return next_race
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding next race: {str(e)}")


def convert_schedule_timezone(schedule, target_timezone):
    """Convert all datetime strings in schedule from UTC to target timezone"""
    try:
        target_tz = pytz.timezone(target_timezone)
        utc_tz = pytz.UTC

        for race in schedule:
            for session_name, session_info in race['sessions'].items():
                if session_info.get('time') == 'TBC':
                    continue

                # Convert start time
                start_str = session_info.get('start')
                if start_str:
                    try:
                        utc_dt = datetime.fromisoformat(start_str)
                        if utc_dt.tzinfo is None:
                            utc_dt = utc_tz.localize(utc_dt)

                        local_dt = utc_dt.astimezone(target_tz)
                        session_info['start'] = local_dt.isoformat()
                    except Exception as e:
                        print(e)
                        continue

                # Convert end time
                end_str = session_info.get('end')
                if end_str:
                    try:
                        utc_dt = datetime.fromisoformat(end_str)
                        if utc_dt.tzinfo is None:
                            utc_dt = utc_tz.localize(utc_dt)

                        local_dt = utc_dt.astimezone(target_tz)
                        session_info['end'] = local_dt.isoformat()
                    except Exception as e:
                        print(e)
                        continue

        return schedule
    except Exception as e:
        print(e)
        return schedule


def convert_race_timezone(race, target_timezone):
    """Convert datetime strings in a single race from UTC to target timezone"""
    try:
        target_tz = pytz.timezone(target_timezone)
        utc_tz = pytz.UTC

        # Convert race sessions
        for session_name, session_info in race['sessions'].items():
            if session_info.get('time') == 'TBC':
                continue

            # Convert start time
            start_str = session_info.get('start')
            if start_str:
                try:
                    utc_dt = datetime.fromisoformat(start_str)
                    if utc_dt.tzinfo is None:
                        utc_dt = utc_tz.localize(utc_dt)

                    local_dt = utc_dt.astimezone(target_tz)
                    session_info['start'] = local_dt.isoformat()
                except Exception as e:
                    print(e)
                    continue

            # Convert end time
            end_str = session_info.get('end')
            if end_str:
                try:
                    utc_dt = datetime.fromisoformat(end_str)
                    if utc_dt.tzinfo is None:
                        utc_dt = utc_tz.localize(utc_dt)

                    local_dt = utc_dt.astimezone(target_tz)
                    session_info['end'] = local_dt.isoformat()
                except Exception as e:
                    print(e)
                    continue

        # Convert next session if exists
        if 'nextSession' in race:
            start_str = race['nextSession'].get('date')
            if start_str:
                try:
                    utc_dt = datetime.fromisoformat(start_str)
                    if utc_dt.tzinfo is None:
                        utc_dt = utc_tz.localize(utc_dt)

                    local_dt = utc_dt.astimezone(target_tz)
                    race['nextSession']['date'] = local_dt.isoformat()
                except Exception as e:
                    print(e)
                    pass

        return race
    except Exception as e:
        print(e)
        return race


# import psutil
# def log_mem_usage():
#    process = psutil.Process(os.getpid())
#    print(f"Memory usage (RSS): {process.memory_info().rss / 1024**2:.1f} MiB")

# if __name__ == "__main__":
#    log_mem_usage()
#    port = int(os.environ.get("PORT", 8000))
#    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", reload=False)
