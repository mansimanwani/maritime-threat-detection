"""The feedback-driven recalibration loop.

Analyst verdicts do not retrain the network. They move the two knobs that
sit between the network's raw errors and the decision to flag:

  feature_weights       nudged toward features that drove confirmed hits
                        and away from features that drove false alarms
  threshold_percentile  raised when precision is below target, lowered
                        when it is above

Both updates are damped by how much feedback exists, so five clicks move
the system a little and fifty move it a lot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import (
    FEATURE_COLUMNS,
    FEEDBACK_PRIOR,
    RECALIBRATION_RATE,
    TARGET_PRECISION,
    THRESHOLD_BOUNDS,
    THRESHOLD_STEP,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from backend.core.db import fetch_feedback, replace_anomalies
from backend.core.detect import detect, load_calibration, save_calibration
from backend.core.explain import explain


def _mean_shares(feedback: pd.DataFrame, label: str) -> dict[str, float]:
    """Average share of the score each feature carried, over one verdict class."""
    rows = feedback[feedback["label"] == label]
    if rows.empty:
        return {f: 0.0 for f in FEATURE_COLUMNS}
    return {f: float(np.mean([d.get(f, 0.0) for d in rows["drivers"]])) for f in FEATURE_COLUMNS}


def recalibrate(feedback: pd.DataFrame, calib: dict) -> tuple[dict, dict]:
    """Return (new calibration, a report of what changed and why)."""
    n = len(feedback)
    if n == 0:
        return calib, {"n_feedback": 0, "note": "no feedback yet, nothing to learn from"}

    n_tp = int((feedback["label"] == "true_positive").sum())
    n_fp = int((feedback["label"] == "false_positive").sum())
    precision = n_tp / n

    # Damping: a handful of verdicts should nudge, not overturn.
    confidence = n / (n + FEEDBACK_PRIOR)

    tp_share = _mean_shares(feedback, "true_positive")
    fp_share = _mean_shares(feedback, "false_positive")

    new_weights, evidence = {}, {}
    for f in FEATURE_COLUMNS:
        # Positive when a feature drove real hits, negative when it drove false alarms.
        ev = tp_share[f] - fp_share[f]
        evidence[f] = ev
        w = calib["feature_weights"][f] * (1.0 + RECALIBRATION_RATE * confidence * ev)
        new_weights[f] = float(np.clip(w, WEIGHT_MIN, WEIGHT_MAX))

    # Keep the average weight at 1.0 so scores stay comparable between versions.
    scale = np.mean(list(new_weights.values()))
    new_weights = {f: float(w / scale) for f, w in new_weights.items()}

    # Too many false alarms -> flag less. Too few -> flag more.
    shift = THRESHOLD_STEP * confidence * (TARGET_PRECISION - precision)
    new_pct = float(
        np.clip(calib["threshold_percentile"] + shift, THRESHOLD_BOUNDS[0], THRESHOLD_BOUNDS[1])
    )

    new_calib = {
        "feature_weights": new_weights,
        "threshold_percentile": new_pct,
        "version": calib["version"] + 1,
        "feedback_used": n,
    }
    report = {
        "n_feedback": n,
        "n_true_positive": n_tp,
        "n_false_positive": n_fp,
        "precision": precision,
        "confidence": confidence,
        "evidence": evidence,
        "old_weights": calib["feature_weights"],
        "new_weights": new_weights,
        "old_percentile": calib["threshold_percentile"],
        "new_percentile": new_pct,
    }
    return new_calib, report


def rescore_and_store(calib: dict) -> int:
    """Re-run detection and explanation under a calibration, refresh the database."""
    scored, _ = detect(calib)
    anomalies = explain(scored)
    replace_anomalies(anomalies, calib["version"])
    return len(anomalies)


def main() -> dict:
    calib = load_calibration()
    feedback = fetch_feedback()
    new_calib, report = recalibrate(feedback, calib)

    if report["n_feedback"] == 0:
        print("No feedback on record. Nothing to recalibrate.")
        return calib

    print(f"feedback         {report['n_feedback']} verdicts "
          f"({report['n_true_positive']} confirmed, {report['n_false_positive']} false alarms)")
    print(f"precision        {report['precision']:.0%}  (target {TARGET_PRECISION:.0%})")
    print(f"confidence       {report['confidence']:.2f}  (damping factor)\n")

    print(f"{'feature':<16}{'evidence':>10}{'weight':>10}{'-->':>6}{'':>8}")
    for f in FEATURE_COLUMNS:
        print(f"{f:<16}{report['evidence'][f]:>+10.2f}"
              f"{report['old_weights'][f]:>10.2f}{'-->':>6}{report['new_weights'][f]:>8.2f}")

    print(f"\nthreshold        {report['old_percentile']:.1f}th --> {report['new_percentile']:.1f}th percentile")

    save_calibration(new_calib)
    n = rescore_and_store(new_calib)
    print(f"calibration      version {calib['version']} --> {new_calib['version']}")
    print(f"re-scored        {n} anomalies now flagged")
    return new_calib


if __name__ == "__main__":
    main()