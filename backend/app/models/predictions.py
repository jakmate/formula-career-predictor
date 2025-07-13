from datetime import date
from typing import Dict, List, Optional
from pydantic import BaseModel

from app.models.system import SystemStatus


class PredictionResponse(BaseModel):
    driver: str
    nationality: Optional[str] = None
    position: int
    avg_finish_pos: float
    std_finish_pos: float
    avg_quali_pos: Optional[float] = None
    std_quali_pos: Optional[float] = None
    points: float
    wins: int
    podiums: int
    win_rate: float
    podium_rate: float
    top_10_rate: float
    dnf_rate: float
    experience: int
    age: Optional[float]
    dob: Optional[date] = None
    participation_rate: float
    pole_rate: float
    top_10_starts_rate: float
    teammate_h2h: float
    team: str
    team_pos: int
    team_points: float
    points_share: float
    raw_probability: float
    empirical_percentage: float


class ModelResults(BaseModel):
    model_name: str
    predictions: List[PredictionResponse]
    accuracy_metrics: Dict[str, float]


class AllPredictionsResponse(BaseModel):
    models: List[str]
    predictions: Dict[str, ModelResults]
    system_status: SystemStatus
