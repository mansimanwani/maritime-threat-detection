"""Explain why a node was flagged, in words an analyst can read.

The score is a weighted sum of five per-feature reconstruction errors, so
each feature's share of that sum is literally how much it drove the flag.
We take the features with the biggest shares, compare their actual values
against what the rest of the fleet was doing, and write a sentence.

This is attribution by decomposition of the score itself -- not a post-hoc
approximation of it. The percentages add up to 100 by construction.
"""

from __future__ import annotations

import json

import pandas as pd

from backend.config import (
    ANOMALIES_CSV,
    EXPLAIN_MIN_SHARE,
    EXPLAIN_TOP_K,
    FEATURE_COLUMNS,
    FEATURES_CSV,
    SCORES_CSV,
    ensure_dirs,
)
from backend.core.detect import load_calibration

# How to say each feature out loud. "high" and "low" are used only when the
# value is genuinely at the edge of the fleet's range; otherwise we use the
# neutral wording, because a value can sit near the fleet median and still be
# wrong *for its own neighbourhood* -- which is exactly what a GNN detects.
PHRASES = {
    "mean_speed": {
        "high": "it was moving unusually fast at {v:.1f} kn",
        "low": "it slowed almost to a stop at {v:.1f} kn",
        "mid": "its average speed of {v:.1f} kn did not fit its neighbourhood",
    },
    "speed_change": {
        "high": "its speed jumped by {v:.1f} kn between pings",
        "low": "its speed was unusually steady",
        "mid": "its speed variation of {v:.1f} kn did not fit its neighbourhood",
    },
    "course_change": {
        "high": "it turned sharply through {v:.0f} degrees",
        "low": "it held an unusually rigid course",
        "mid": "its {v:.0f}-degree course change did not fit its neighbourhood",
    },
    "gap_minutes": {
        "high": "it stopped transmitting for {v:.0f} minutes",
        "low": "it was reporting unusually often",
        "mid": "its reporting interval of {v:.0f} min did not fit its neighbourhood",
    },
    "neighbor_count": {
        "high": "it had {v:.0f} other vessel(s) close alongside",
        "low": "it was unusually isolated",
        "mid": "its {v:.0f} nearby vessel(s) did not fit its neighbourhood",
    },
}

TYPICAL = "typical is {t}"


def _fmt_typical(feature: str, value: float) -> str:
    return f"{value:.0f}" if feature in ("course_change", "gap_minutes") else f"{value:.1f}"


def fleet_baseline(feats: pd.DataFrame) -> pd.DataFrame:
    """Median plus the 5th/95th percentiles, used to choose wording."""
    return feats[FEATURE_COLUMNS].quantile([0.05, 0.50, 0.95])


def contributions(row: pd.Series, weights: dict) -> dict[str, float]:
    """Each feature's share of the score, as a fraction summing to 1."""
    raw = {f: max(row[f"err_{f}"], 0.0) * max(weights[f], 0.0) for f in FEATURE_COLUMNS}
    total = sum(raw.values())
    if total <= 0:
        return {f: 0.0 for f in FEATURE_COLUMNS}
    return {f: v / total for f, v in raw.items()}


def _pick_drivers(shares: dict[str, float]) -> list[str]:
    ranked = sorted(shares, key=shares.get, reverse=True)
    picked = [ranked[0]]
    for f in ranked[1:EXPLAIN_TOP_K]:
        if shares[f] >= EXPLAIN_MIN_SHARE:
            picked.append(f)
    return picked


def build_reason(row: pd.Series, shares: dict[str, float], baseline: pd.DataFrame) -> str:
    parts = []
    for feature in _pick_drivers(shares):
        value = row[feature]
        low, typical, high = baseline[feature].loc[[0.05, 0.50, 0.95]]

        if value > high:
            direction = "high"
        elif value < low:
            direction = "low"
        else:
            direction = "mid"

        phrase = PHRASES[feature][direction].format(v=value)
        parts.append(
            f"{phrase} ({TYPICAL.format(t=_fmt_typical(feature, typical))}, "
            f"{shares[feature]:.0%} of the score)"
        )

    joined = parts[0] if len(parts) == 1 else " and ".join(parts)
    return f"Flagged because {joined}."


def explain(scored: pd.DataFrame | None = None) -> pd.DataFrame:
    if scored is None:
        if not SCORES_CSV.exists():
            raise FileNotFoundError(
                f"No scores at {SCORES_CSV}. Run: python -m backend.core.detect"
            )
        scored = pd.read_csv(SCORES_CSV)

    feats = pd.read_csv(FEATURES_CSV)
    baseline = fleet_baseline(feats)
    weights = load_calibration()["feature_weights"]

    merged = scored.merge(feats[["window_id", "mmsi"] + FEATURE_COLUMNS], on=["window_id", "mmsi"])
    flagged = merged[merged["is_anomaly"]].sort_values("score", ascending=False).reset_index(drop=True)

    reasons, driver_json = [], []
    for _, row in flagged.iterrows():
        shares = contributions(row, weights)
        reasons.append(build_reason(row, shares, baseline))
        driver_json.append(json.dumps({f: round(s, 4) for f, s in shares.items()}))

    out = flagged[["window_id", "window_start", "mmsi", "lat", "lon", "score"]].copy()
    out["reason"] = reasons
    out["drivers"] = driver_json
    return out


def main() -> pd.DataFrame:
    ensure_dirs()
    anomalies = explain()
    anomalies.to_csv(ANOMALIES_CSV, index=False)

    print(f"{len(anomalies)} anomalies explained\n")
    for _, r in anomalies.head(6).iterrows():
        print(f"[window {r.window_id}] vessel {r.mmsi}  score {r.score:.2f}")
        print(f"  {r.reason}\n")
    print(f"wrote -> {ANOMALIES_CSV}")
    return anomalies


if __name__ == "__main__":
    main()