from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: Dict[str, int]
    last_training: Optional[datetime]


class SystemStatus(BaseModel):
    last_scrape: Optional[datetime]
    last_training: Optional[datetime]
    models_available: Dict[str, List[str]]
    data_health: Dict[str, Dict[str, int]]


class RefreshResponse(BaseModel):
    message: str
    task_id: Optional[str] = None
    estimated_completion: Optional[datetime] = None
