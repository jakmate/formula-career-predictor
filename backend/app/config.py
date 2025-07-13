from datetime import datetime
import logging
import os
from pathlib import Path

CURRENT_YEAR = datetime.now().year
SEASON_END_MONTH = 12

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
STATE_FILE = BASE_DIR / 'system_state.json'
SCHEDULE_DIR = BASE_DIR / 'data' / 'schedules' / str(CURRENT_YEAR)

PORT = int(os.environ.get("PORT", 8000))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
