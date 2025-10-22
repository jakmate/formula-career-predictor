from typing import Optional
from app.services.data_service import DataService
from app.core.state import AppState
from app.services.model_service import ModelService
from app.services.cronjobs_service import CronjobService
from app.config import LOGGER

# Global application state
app_state: Optional[AppState] = None
model_service: Optional[ModelService] = None
data_service: Optional[DataService] = None
scheduler_service: Optional[CronjobService] = None

# Global data cache
data_cache = {}


async def initialize_app_state():
    """Initialize application state and services"""
    global app_state, model_service, data_service, scheduler_service

    # Initialize state
    app_state = AppState()
    app_state.load_state()

    # Initialize services
    model_service = ModelService(app_state)
    data_service = DataService(app_state, data_cache)
    scheduler_service = CronjobService(app_state, model_service, data_service)

    # Try to load pre-trained models
    if not await model_service.load_models():
        LOGGER.info("No models found. Initializing system...")
        await data_service.initialize_system()

    # Start scheduler
    await scheduler_service.start()


async def cleanup_app_state():
    """Clean up application state and services"""
    if scheduler_service:
        await scheduler_service.stop()
    if app_state:
        app_state.save_state()


def get_app_state() -> AppState:
    """Get application state dependency"""
    if app_state is None:
        raise RuntimeError("Application state not initialized")
    return app_state


def get_model_service() -> ModelService:
    """Get model service dependency"""
    if model_service is None:
        raise RuntimeError("Model service not initialized")
    return model_service


def get_data_service() -> DataService:
    """Get data service dependency"""
    if data_service is None:
        raise RuntimeError("Data service not initialized")
    return data_service


def get_scheduler_service() -> CronjobService:
    """Get scheduler service dependency"""
    if scheduler_service is None:
        raise RuntimeError("Scheduler service not initialized")
    return scheduler_service
