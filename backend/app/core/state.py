import json
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

from app.config import LOGGER, STATE_FILE


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
            LOGGER.error(f"Corrupted state file: {e}. Reinitializing state.")
            # Backup corrupted file
            os.rename(STATE_FILE, f"{STATE_FILE}.backup")
            return False

        except Exception as e:
            LOGGER.error(f"Error loading state: {e}")

        return False
