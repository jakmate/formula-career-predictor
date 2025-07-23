from datetime import datetime
import logging
import os
from pathlib import Path

CURRENT_YEAR = datetime.now().year
SEASON_END_MONTH = 12
SEED = 69
NOT_PARTICIPATED_CODES = ['nan', 'DNS', 'WD', 'DNQ', 'DNA', 'C', 'EX']
RETIREMENT_CODES = ['Ret', 'NC', 'DSQ', 'DSQP']

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
STATE_FILE = BASE_DIR / 'system_state.json'
DATA_DIR = BASE_DIR / "data"
SCHEDULE_DIR = DATA_DIR / 'schedules' / str(CURRENT_YEAR)
PROFILES_DIR = DATA_DIR / 'driver_profiles'

PORT = int(os.environ.get("PORT", 8000))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
