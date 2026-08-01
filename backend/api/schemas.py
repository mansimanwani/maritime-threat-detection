"""Request and response shapes for the API.

These exist so the contract between Python and React is written down in
one place, and so FastAPI can reject malformed requests before they reach
any of our code.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AnomalyOut(BaseModel):
    id: int
    window_id: int
    window_start: str
    mmsi: int
    lat: float
    lon: float
    score: float
    reason: str
    drivers: dict[str, float]
    calibration_version: int
    feedback_label: str | None = None


class TrackPoint(BaseModel):
    timestamp: str
    lat: float
    lon: float
    sog: float


class FeedbackIn(BaseModel):
    window_id: int
    mmsi: int
    label: Literal["true_positive", "false_positive"]


class FeedbackOut(BaseModel):
    id: int
    recorded: bool = True
    total_feedback: int


class CalibrationOut(BaseModel):
    version: int
    feedback_used: int
    threshold_percentile: float
    feature_weights: dict[str, float]


class StatusOut(BaseModel):
    calibration: CalibrationOut
    n_anomalies: int
    n_feedback: int


class RecalibrateOut(BaseModel):
    ran: bool
    n_feedback: int
    precision: float | None = None
    old_version: int
    new_version: int
    old_percentile: float | None = None
    new_percentile: float | None = None
    old_weights: dict[str, float] | None = None
    new_weights: dict[str, float] | None = None
    n_anomalies: int
    message: str = ""