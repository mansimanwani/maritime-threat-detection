"""Turn cleaned AIS pings into behavioural features, one row per vessel per time window.

A single ping says nothing suspicious on its own. Behaviour only appears
over a stretch of time, so we slice the day into fixed windows and
describe what each vessel did inside each one.

Output: data/processed/features.csv with the five columns listed in
config.FEATURE_COLUMNS, plus the metadata the map and the graph builder need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import (
    CLEAN_PINGS_CSV,
    FEATURES_CSV,
    FEATURE_COLUMNS,
    PROXIMITY_KM,
    WINDOW_MINUTES,
    ensure_dirs,
)

EARTH_RADIUS_KM = 6371.0

META_COLUMNS = ["window_id", "window_start", "mmsi", "lat", "lon", "n_pings"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _circular_diff(series: pd.Series) -> pd.Series:
    """Smallest angle between consecutive compass headings, in degrees."""
    d = series.diff().abs()
    return np.minimum(d, 360.0 - d)


def _add_neighbor_count(windows: pd.DataFrame) -> pd.DataFrame:
    """For each window, count other vessels within PROXIMITY_KM."""
    counts = np.zeros(len(windows), dtype=int)

    for _, idx in windows.groupby("window_start").groups.items():
        pos = windows.loc[idx]
        lat = pos["lat"].to_numpy()
        lon = pos["lon"].to_numpy()

        dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        np.fill_diagonal(dist, np.inf)          # a vessel is not its own neighbour
        counts[windows.index.get_indexer(idx)] = (dist < PROXIMITY_KM).sum(axis=1)

    windows = windows.copy()
    windows["neighbor_count"] = counts
    return windows


def build_features(pings: pd.DataFrame) -> pd.DataFrame:
    pings = pings.sort_values(["mmsi", "timestamp"]).copy()
    pings["timestamp"] = pd.to_datetime(pings["timestamp"], utc=True)

    # Per-vessel change between consecutive pings. Computed on the full
    # track first, so a gap that straddles a window boundary is still seen.
    grp = pings.groupby("mmsi", sort=False)
    pings["gap_min"] = grp["timestamp"].diff().dt.total_seconds().div(60).fillna(0.0)
    pings["speed_diff"] = grp["sog"].diff().abs().fillna(0.0)
    pings["course_diff"] = grp["cog"].transform(_circular_diff).fillna(0.0)

    pings["window_start"] = pings["timestamp"].dt.floor(f"{WINDOW_MINUTES}min")

    windows = (
        pings.groupby(["mmsi", "window_start"], as_index=False)
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            n_pings=("sog", "size"),
            mean_speed=("sog", "mean"),
            speed_change=("speed_diff", "max"),
            course_change=("course_diff", "max"),
            gap_minutes=("gap_min", "max"),
        )
    )

    windows = _add_neighbor_count(windows)

    starts = sorted(windows["window_start"].unique())
    window_id = {t: i for i, t in enumerate(starts)}
    windows["window_id"] = windows["window_start"].map(window_id)

    windows = windows[META_COLUMNS + FEATURE_COLUMNS]
    return windows.sort_values(["window_id", "mmsi"]).reset_index(drop=True)


def main() -> pd.DataFrame:
    ensure_dirs()
    if not CLEAN_PINGS_CSV.exists():
        raise FileNotFoundError(
            f"No clean pings at {CLEAN_PINGS_CSV}. Run: python -m backend.core.ingest"
        )

    pings = pd.read_csv(CLEAN_PINGS_CSV)
    feats = build_features(pings)
    feats.to_csv(FEATURES_CSV, index=False)

    print(f"{len(feats):,} rows  =  {feats['mmsi'].nunique()} vessels x {feats['window_id'].nunique()} windows")
    print(f"window length: {WINDOW_MINUTES} min\n")
    print(feats[FEATURE_COLUMNS].describe().round(2).to_string())
    print(f"\nwrote -> {FEATURES_CSV}")
    return feats


if __name__ == "__main__":
    main()