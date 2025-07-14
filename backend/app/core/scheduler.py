import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.state import AppState
from app.config import CURRENT_YEAR, LOGGER, SEASON_END_MONTH
from app.core.scraping.scrape import scrape_current_year


class SchedulerService:
    def __init__(self, app_state: AppState, model_service, data_service):
        self.app_state = app_state
        self.model_service = model_service
        self.data_service = data_service
        self.scheduler = AsyncIOScheduler()

    async def start(self):
        """Start scheduler"""
        self.scheduler.add_job(
            self.scrape_and_train_task,
            'cron', day_of_week='mon', hour=3,
            id='weekly_scrape_train'
        )
        self.scheduler.start()
        LOGGER.info("Scheduler started")

    async def stop(self):
        """Stop scheduler"""
        self.scheduler.shutdown()

    async def scrape_and_train_task(self):
        """Combined scraping and training task for new seasons"""
        try:
            LOGGER.info("Starting data scraping task...")
            await asyncio.get_event_loop().run_in_executor(None, scrape_current_year)
            self.app_state.system_status["last_scrape"] = datetime.now()
            LOGGER.info("Data scraping completed")

            if (self._is_season_complete() and CURRENT_YEAR >
                    self.app_state.system_status["last_trained_season"]):
                LOGGER.info(f"New season {CURRENT_YEAR} complete. Starting training...")
                await self._train_models_task()
            else:
                LOGGER.info("No new complete season available. Updating predictions only.")
                from app.services.prediction_service import PredictionService
                prediction_service = PredictionService(self.app_state)
                await prediction_service.update_predictions()
        except Exception as e:
            LOGGER.error(f"Scrape and train task failed: {e}")
        finally:
            self.app_state.save_state()

    async def _train_models_task(self):
        """Train models on newly available complete seasons"""
        try:
            await self.data_service.initialize_system()
        except Exception as e:
            LOGGER.error(f"Training task failed: {e}")

    def _is_season_complete(self):
        """Check if current season is complete based on date"""
        now = datetime.now()
        return now.month > SEASON_END_MONTH and now.year == CURRENT_YEAR
