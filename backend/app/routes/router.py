from fastapi import APIRouter
import app.routes.health as health
import app.routes.predictions as predictions
import app.routes.schedule as schedule
import app.routes.system as system

api_router = APIRouter()
api_router.include_router(health.router, prefix="", tags=["Health"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
api_router.include_router(schedule.router, prefix="/races", tags=["Schedule"])
api_router.include_router(system.router, prefix="/system", tags=["System"])
