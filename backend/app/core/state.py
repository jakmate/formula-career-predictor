import json
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

from app.config import LOGGER, STATE_FILE


class AppState:
    def __init__(self):
        self.models = {
            'f3_to_f2': {},
            'f2_to_f1': {}
        }
        self.feature_cols = {
            'f3_to_f2': [],
            'f2_to_f1': []
        }
        self.scaler = {
            'f3_to_f2': None,
            'f2_to_f1': None
        }
        self.current_predictions = {}
        self.system_status = {
            "last_scrape_full": None,
            "last_scrape_predictions": None,
            "last_scrape_schedule": None,
            "last_training": None,
            "last_trained_season": None,
            "models_available": {
                "f3_to_f2": [],
                "f2_to_f1": []
            },
            "data_health": {}
        }
        self.scheduler = AsyncIOScheduler()

    def save_state(self):
        """Save critical state to disk"""
        state = {
            "last_scrape_full": self.system_status["last_scrape_full"].isoformat()
            if self.system_status["last_scrape_full"] else None,
            "last_scrape_predictions": self.system_status["last_scrape_predictions"].isoformat()
            if self.system_status["last_scrape_predictions"] else None,
            "last_scrape_schedule": self.system_status["last_scrape_schedule"].isoformat()
            if self.system_status["last_scrape_schedule"] else None,
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

                self.system_status["last_scrape_full"] = (
                    datetime.fromisoformat(state["last_scrape_full"])
                    if state.get("last_scrape_full") else None
                )
                self.system_status["last_training"] = (
                    datetime.fromisoformat(state["last_training"])
                    if state["last_training"] else None
                )
                self.system_status["last_trained_season"] = state["last_trained_season"]
                loaded_models_avail = state.get("models_available", {})
                if not isinstance(loaded_models_avail, dict):
                    LOGGER.warning("models_available is not a dict. Resetting to default.")
                    loaded_models_avail = {"f3_to_f2": [], "f2_to_f1": []}
                else:
                    # Ensure both keys exist
                    for key in ["f3_to_f2", "f2_to_f1"]:
                        if key not in loaded_models_avail:
                            loaded_models_avail[key] = []

                self.system_status["models_available"] = loaded_models_avail

                return True

        except json.JSONDecodeError as e:
            LOGGER.error(f"Corrupted state file: {e}. Reinitializing state.")
            # Backup corrupted file
            os.rename(STATE_FILE, f"{STATE_FILE}.backup")
            return False

        except Exception as e:
            LOGGER.error(f"Error loading state: {e}")

        return False
