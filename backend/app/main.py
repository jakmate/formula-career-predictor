import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import LOGGER
from app.dependencies import initialize_app_state, cleanup_app_state
from app.routes.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown handling"""
    try:
        LOGGER.info("Starting application...")

        # Initialize application state
        await initialize_app_state()

        yield
    finally:
        LOGGER.info("Shutting down application...")
        await cleanup_app_state()
        LOGGER.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations"""
    app = FastAPI(
        title="Formula Predictions API",
        version="1.0.0",
        description="API for predicting Formula 1, 2, 3 and career promotions",
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

    # Include API routes
    app.include_router(api_router, prefix="/api")

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug", reload=False)
