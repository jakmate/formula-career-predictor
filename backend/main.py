import asyncio
import logging
import pandas as pd
import torch
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional


from scraper import scrape
from predictor import (
    load_and_combine_data,
    create_target_using_f2_data,
    calculate_teammate_performance,
    calculate_years_in_f3_combined,
    engineer_features,
    train_models,
    predict_drivers
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="F3/F2 Racing Predictions API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionResponse(BaseModel):
    driver: str
    position: int
    points: float
    win_rate: float
    podium_rate: float
    top_10_rate: float
    dnf_rate: float
    points_per_race: float
    experience: int
    age: Optional[float]
    has_academy: int
    h2h_rate: float
    avg_pos_diff: float
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

# Global variables to store models and data
traditional_models = {}
deep_models = {}
feature_cols = []
scaler = None
system_status = {
    "last_scrape": None,
    "last_training": None,
    "models_available": [],
    "data_health": {}
}

# Scheduler for periodic tasks
scheduler = AsyncIOScheduler()

async def scrape_data_task():
    """Background task to scrape racing data"""
    try:
        logger.info("Starting data scraping task...")
        await asyncio.get_event_loop().run_in_executor(None, scrape)
        system_status["last_scrape"] = datetime.now()
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
        
        # Process data using the correct function names
        f3_df = calculate_teammate_performance(f3_df)
        f3_df = calculate_years_in_f3_combined(f3_df)
        f3_df = create_target_using_f2_data(f3_df, f2_df)
        
        features_df = engineer_features(f3_df)
        features_df['moved_to_f2'] = f3_df['moved_to_f2']
        
        # Train models using the unified function
        global traditional_models, deep_models, feature_cols, scaler
        traditional_models, deep_models, _, _, feature_cols, scaler = train_models(features_df)
        
        # Update system status
        system_status["last_training"] = datetime.now()
        system_status["models_available"] = list(traditional_models.keys()) + list(deep_models.keys())
        system_status["data_health"] = {
            "f3_records": len(f3_df),
            "f2_records": len(f2_df),
            "features_records": len(features_df)
        }
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting F3/F2 Predictions API...")
    
    # Schedule tasks
    scheduler.add_job(
        scrape_data_task,
        CronTrigger(day_of_week='fri', hour=9, minute=0),
        id='scrape_data',
        replace_existing=True
    )
    
    scheduler.add_job(
        train_models_task,
        CronTrigger(day_of_week='fri', hour=10, minute=0),
        id='train_models',
        replace_existing=True
    )
    
    scheduler.start()
    
    # Load existing models if available
    await train_models_task()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    scheduler.shutdown()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "F3/F2 Racing Predictions API is running"}

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status information"""
    return SystemStatus(**system_status)

@app.post("/scrape")
async def trigger_scrape(background_tasks: BackgroundTasks):
    """Manually trigger data scraping"""
    background_tasks.add_task(scrape_data_task)
    return {"message": "Data scraping started in background"}

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """Manually trigger model training"""
    background_tasks.add_task(train_models_task)
    return {"message": "Model training started in background"}

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "traditional_models": list(traditional_models.keys()),
        "deep_learning_models": list(deep_models.keys()),
        "total_models": len(traditional_models) + len(deep_models)
    }

@app.get("/predictions/{model_name}")
async def get_predictions(model_name: str) -> ModelResults:
    """Get predictions from a specific model"""
    if model_name not in traditional_models and model_name not in deep_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Load current F3 data
        f3_df = load_and_combine_data('F3')
        f2_df = load_and_combine_data('F2')
        
        if f3_df.empty:
            raise HTTPException(status_code=404, detail="No F3 data available")
        
        # Process data
        f3_df = calculate_teammate_performance(f3_df)
        f3_df = calculate_years_in_f3_combined(f3_df)
        f3_df = create_target_using_f2_data(f3_df, f2_df)
        
        features_df = engineer_features(f3_df)
        features_df['moved_to_f2'] = f3_df['moved_to_f2']
        
        # Get 2025 drivers
        current_year = 2025
        f3_current_drivers = features_df[features_df['year'] == current_year].copy()
        if f3_current_drivers.empty:
            current_year = features_df['year'].max()
            f3_current_drivers = features_df[features_df['year'] == current_year].copy()
        
        if f3_current_drivers.empty:
            raise HTTPException(status_code=404, detail="No current drivers data available")
        
        predictions = []
        X_current = f3_current_drivers[feature_cols].fillna(0)
        
        if model_name in traditional_models:
            # Traditional ML model
            model = traditional_models[model_name]
            raw_probas = model.predict_proba(X_current)[:, 1]
            predictions_binary = model.predict(X_current)
            
        elif model_name in deep_models:
            # Deep learning model
            model = deep_models[model_name]
            X_current_scaled = scaler.transform(X_current)
            
            if 'PyTorch' in model_name:
                model.eval()
                with torch.no_grad():
                    X_torch = torch.FloatTensor(X_current_scaled)
                    if torch.cuda.is_available():
                        X_torch = X_torch.cuda()
                    logits = model(X_torch)
                    raw_probas = torch.sigmoid(logits).cpu().numpy().flatten()
            else:  # Keras model
                raw_probas = model.predict(X_current_scaled, verbose=0).flatten()
            
            predictions_binary = (raw_probas > 0.5).astype(int)
        
        # Apply calibration if available
        if hasattr(model, 'calibration_map') and hasattr(model, 'calibration_bins'):
            empirical_pct = []
            for p in raw_probas:
                bin_interval = pd.cut([p], bins=model.calibration_bins, include_lowest=True)[0]
                empirical_pct.append(model.calibration_map.get(bin_interval, p) * 100)
        else:
            empirical_pct = raw_probas * 100
        
        # Create prediction responses
        for idx, (_, row) in enumerate(f3_current_drivers.iterrows()):
            predictions.append(PredictionResponse(
                driver=row['driver'],
                position=int(row['final_position']),
                points=float(row['points']),
                win_rate=float(row['win_rate']),
                podium_rate=float(row['podium_rate']),
                top_10_rate=float(row['top_10_rate']),
                dnf_rate=float(row['dnf_rate']),
                points_per_race=float(row['points_per_race']),
                experience=int(row['years_in_f3']),
                age=float(row['age']) if pd.notna(row['age']) else None,
                has_academy=int(row['has_academy']),
                h2h_rate=float(row['teammate_h2h_rate']),
                avg_pos_diff=float(row['avg_pos_vs_teammates']),
                raw_probability=float(raw_probas[idx]),
                empirical_percentage=float(empirical_pct[idx]),
                prediction=int(predictions_binary[idx])
            ))
        
        # Sort by empirical percentage (highest first)
        predictions.sort(key=lambda x: x.empirical_percentage, reverse=True)
        
        return ModelResults(
            model_name=model_name,
            predictions=predictions,
            accuracy_metrics={"total_predictions": len(predictions)}
        )
        
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
async def get_all_predictions():
    """Get predictions from all available models"""
    all_predictions = {}
    
    for model_name in list(traditional_models.keys()) + list(deep_models.keys()):
        try:
            result = await get_predictions(model_name)
            all_predictions[model_name] = result
        except Exception as e:
            logger.error(f"Error getting predictions for {model_name}: {e}")
            all_predictions[model_name] = {"error": str(e)}
    
    return all_predictions

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)