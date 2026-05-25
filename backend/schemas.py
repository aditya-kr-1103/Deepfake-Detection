from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


class DetectionResponse(BaseModel):
    prediction: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    media_type: str
    frames_used: int | None = None
    message: str | None = None
