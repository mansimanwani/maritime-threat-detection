"""Turn raw reconstruction errors into calibrated anomaly scores and flags.

Two knobs live here, and they are the two things analyst feedback will
later move (see recalibrate.py):

  feature_weights      how much each feature counts toward the score
  threshold_percentile how high a score has to be before we flag it

Both are read from artifacts/calibration.json when it exists, and fall
back to the neutral defaults in config.py when it does not.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from backend.config import (
    CALIBRATION_PATH,
    DEFAULT_FEATURE_WEIGHTS,
    FEATURE_COLUMNS,
    FEATURES_CSV,
    MODEL_PATH,
    SCORES_CSV,
    THRESHOLD_PERCENTILE,
    ensure_dirs,
)
from backend.core.graph_builder import load_graph
from backend.core.model import GNNAutoencoder, reconstruction_error


def load_calibration() -> dict:
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return {
        "feature_weights": dict(DEFAULT_FEATURE_WEIGHTS),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "version": 0,
        "feedback_used": 0,
    }


def save_calibration(calib: dict) -> None:
    ensure_dirs()
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(calib, f, indent=2)


def per_feature_errors(data=None) -> tuple[np.ndarray, "torch.Tensor"]:
    """Run the trained model and return the [nodes x features] error matrix."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run: python -m backend.core.train"
        )
    data = data if data is not None else load_graph()

    model = GNNAutoencoder(in_dim=data.num_node_features)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        x_hat, _ = model(data.x, data.edge_index)
        errors = reconstruction_error(data.x, x_hat)

    return errors.numpy(), data


def score_nodes(errors: np.ndarray, calib: dict) -> np.ndarray:
    """Weighted average of the per-feature errors."""
    weights = np.array([calib["feature_weights"][f] for f in FEATURE_COLUMNS], dtype=float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    return (errors * weights).sum(axis=1) / weights.sum()


def detect(calib: dict | None = None) -> tuple[pd.DataFrame, float]:
    calib = calib or load_calibration()
    errors, data = per_feature_errors()

    nodes = pd.read_csv(FEATURES_CSV).sort_values(["window_id", "mmsi"]).reset_index(drop=True)
    if len(nodes) != len(errors):
        raise ValueError("features.csv and the saved graph disagree; rebuild the graph")

    scores = score_nodes(errors, calib)
    threshold = float(np.percentile(scores, calib["threshold_percentile"]))

    out = nodes[["window_id", "window_start", "mmsi", "lat", "lon"]].copy()
    for i, feat in enumerate(FEATURE_COLUMNS):
        out[f"err_{feat}"] = errors[:, i]
    out["score"] = scores
    out["is_anomaly"] = scores >= threshold

    return out.sort_values("score", ascending=False).reset_index(drop=True), threshold


def main() -> pd.DataFrame:
    ensure_dirs()
    calib = load_calibration()
    scored, threshold = detect(calib)
    scored.to_csv(SCORES_CSV, index=False)

    flagged = scored[scored["is_anomaly"]]
    print(f"calibration      version {calib['version']}, from {calib['feedback_used']} feedback item(s)")
    print(f"feature weights  {calib['feature_weights']}")
    print(f"threshold        {calib['threshold_percentile']}th percentile = {threshold:.3f}")
    print(f"flagged          {len(flagged)} of {len(scored)} nodes ({flagged['mmsi'].nunique()} distinct vessels)\n")
    print(flagged.head(10)[["window_id", "mmsi", "score"]].round(2).to_string(index=False))
    print(f"\nwrote -> {SCORES_CSV}")
    return scored


if __name__ == "__main__":
    main()