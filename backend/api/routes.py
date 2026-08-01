"""HTTP endpoints. Thin wrappers -- all real work lives in backend/core."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.config import CLEAN_PINGS_CSV
from backend.core.db import fetch_anomalies, fetch_feedback, insert_feedback
from backend.core.detect import load_calibration, save_calibration
from backend.core.recalibrate import recalibrate, rescore_and_store
from backend.api.schemas import (
    AnomalyOut,
    CalibrationOut,
    FeedbackIn,
    FeedbackOut,
    RecalibrateOut,
    StatusOut,
    TrackPoint,
)

router = APIRouter(prefix="/api")


@lru_cache(maxsize=1)
def _pings() -> pd.DataFrame:
    if not CLEAN_PINGS_CSV.exists():
        raise HTTPException(503, "No cleaned pings. Run the pipeline first.")
    return pd.read_csv(CLEAN_PINGS_CSV)


def _calibration_out() -> CalibrationOut:
    c = load_calibration()
    return CalibrationOut(
        version=c["version"],
        feedback_used=c["feedback_used"],
        threshold_percentile=c["threshold_percentile"],
        feature_weights=c["feature_weights"],
    )


@router.get("/status", response_model=StatusOut)
def status():
    return StatusOut(
        calibration=_calibration_out(),
        n_anomalies=len(fetch_anomalies()),
        n_feedback=len(fetch_feedback()),
    )


@router.get("/anomalies", response_model=list[AnomalyOut])
def anomalies(limit: int | None = None):
    return [AnomalyOut(**a) for a in fetch_anomalies(limit=limit)]


@router.get("/track/{mmsi}", response_model=list[TrackPoint])
def track(mmsi: int):
    """The vessel's full path, so an anomaly can be judged in context."""
    df = _pings()
    rows = df[df["mmsi"] == mmsi].sort_values("timestamp")
    if rows.empty:
        raise HTTPException(404, f"No track for vessel {mmsi}")

    return [
        TrackPoint(timestamp=str(r.timestamp), lat=float(r.lat), lon=float(r.lon), sog=float(r.sog))
        for r in rows.itertuples()
    ]


@router.post("/feedback", response_model=FeedbackOut)
def feedback(item: FeedbackIn):
    """Record an analyst verdict. Does not recalibrate -- that is a separate call."""
    match = [
        a for a in fetch_anomalies()
        if a["window_id"] == item.window_id and a["mmsi"] == item.mmsi
    ]
    if not match:
        raise HTTPException(404, f"No current anomaly for vessel {item.mmsi} in window {item.window_id}")

    new_id = insert_feedback(item.window_id, item.mmsi, item.label, match[0]["drivers"])
    return FeedbackOut(id=new_id, total_feedback=len(fetch_feedback()))


@router.post("/recalibrate", response_model=RecalibrateOut)
def run_recalibration():
    """Close the loop: learn from verdicts, re-score, refresh the anomaly list."""
    calib = load_calibration()
    fb = fetch_feedback()
    new_calib, report = recalibrate(fb, calib)

    if report["n_feedback"] == 0:
        return RecalibrateOut(
            ran=False,
            n_feedback=0,
            old_version=calib["version"],
            new_version=calib["version"],
            n_anomalies=len(fetch_anomalies()),
            message="No feedback on record yet.",
        )

    save_calibration(new_calib)
    n = rescore_and_store(new_calib)

    return RecalibrateOut(
        ran=True,
        n_feedback=report["n_feedback"],
        precision=report["precision"],
        old_version=calib["version"],
        new_version=new_calib["version"],
        old_percentile=report["old_percentile"],
        new_percentile=report["new_percentile"],
        old_weights=report["old_weights"],
        new_weights=report["new_weights"],
        n_anomalies=n,
        message=f"Recalibrated from {report['n_feedback']} verdicts.",
    )