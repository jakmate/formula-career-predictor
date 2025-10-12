from datetime import date
from typing import Dict, List, Optional
from pydantic import BaseModel

from app.models.system import SystemStatus


class PredictionResponse(BaseModel):
    driver: str
    nationality: Optional[str] = None
    position: int
    points: float
    wins: int
    podiums: int
    win_rate: float
    dnf_rate: float
    experience: int
    age: Optional[float]
    dob: Optional[date] = None
    participation_rate: float
    teammate_h2h: float
    team: str
    team_pos: int
    team_points: float
    empirical_percentage: Optional[float] = None


class ModelResults(BaseModel):
    model_name: str
    predictions: List[PredictionResponse]
    accuracy_metrics: Dict[str, float]


class PredictionsResponse(BaseModel):
    models: List[str]
    predictions: Dict[str, ModelResults]
    system_status: SystemStatus
