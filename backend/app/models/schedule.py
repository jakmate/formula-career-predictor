from pydantic import BaseModel, Field
from typing import Optional, Literal


class ScheduleRequest(BaseModel):
    series: Literal['f1', 'f2', 'f3'] = Field(..., description="Racing series")
    timezone: Optional[str] = Field(None, description="Target timezone for conversion")
    x_timezone: Optional[str] = Field(None, description="Alternative timezone header")

    def get_timezone(self) -> str:
        """Get the effective timezone (timezone takes precedence over x_timezone)"""
        return self.timezone or self.x_timezone or 'UTC'
