"""Generate a small synthetic AIS dataset using the real AIS field schema.

Produces data/raw/ais_sample.csv with columns:
    mmsi, timestamp, lat, lon, sog, cog

A handful of vessels are given deliberately odd behaviour (an AIS gap,
a loitering rendezvous pair, an erratic manoeuvrer) so that we have
something for the detector to find. Their MMSIs are written to
data/raw/planted_anomalies.txt for our own checking only -- nothing in
the pipeline ever reads that file.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from backend.config import RAW_AIS_CSV, RAW_DIR, RANDOM_SEED, ensure_dirs

# ---------- Scenario settings ----------
N_NORMAL = 36
DURATION_HOURS = 6
PING_SECONDS = 120
START_TIME = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)

# A box in the Gulf of Aden -- a real shipping corridor.
LAT_MIN, LAT_MAX = 12.0, 13.5
LON_MIN, LON_MAX = 44.0, 46.0

NM_PER_DEGREE = 60.0


def _steps() -> int:
    return int(DURATION_HOURS * 3600 / PING_SECONDS)


def _advance(lat: float, lon: float, sog: float, heading_deg: float):
    """Move a point by `sog` knots held for one ping interval."""
    hours = PING_SECONDS / 3600.0
    dist_deg = (sog * hours) / NM_PER_DEGREE
    rad = math.radians(heading_deg)
    dlat = dist_deg * math.cos(rad)
    dlon = dist_deg * math.sin(rad) / max(math.cos(math.radians(lat)), 0.1)
    return lat + dlat, lon + dlon


def _straight_track(rng: random.Random) -> list[tuple[float, float]]:
    """A vessel transiting the corridor at a steady speed."""
    lat = rng.uniform(LAT_MIN, LAT_MAX)
    lon = rng.uniform(LON_MIN, LON_MAX)
    heading = rng.choice([80.0, 260.0]) + rng.uniform(-12, 12)
    speed = rng.uniform(9.0, 16.0)

    track = [(lat, lon)]
    for _ in range(_steps() - 1):
        heading += rng.gauss(0, 1.5)
        lat, lon = _advance(lat, lon, speed, heading)
        track.append((lat, lon))
    return track


def _erratic_track(rng: random.Random) -> list[tuple[float, float]]:
    """Sharp turns and large speed swings."""
    lat = rng.uniform(LAT_MIN, LAT_MAX)
    lon = rng.uniform(LON_MIN, LON_MAX)
    heading = rng.uniform(0, 360)

    track = [(lat, lon)]
    for i in range(_steps() - 1):
        if i % 4 == 0:
            heading += rng.choice([-1, 1]) * rng.uniform(50, 110)
        speed = rng.uniform(1.5, 20.0)
        lat, lon = _advance(lat, lon, speed, heading)
        track.append((lat, lon))
    return track


def _rendezvous_tracks(rng: random.Random):
    """Two vessels converge, sit together, then separate."""
    n = _steps()
    approach = int(n * 0.35)
    loiter = int(n * 0.30)
    depart = n - approach - loiter

    meet_lat = rng.uniform(LAT_MIN + 0.3, LAT_MAX - 0.3)
    meet_lon = rng.uniform(LON_MIN + 0.3, LON_MAX - 0.3)

    tracks = []
    for side in (-1, 1):
        start_lat = meet_lat + side * 0.45
        start_lon = meet_lon - side * 0.55
        end_lat = meet_lat - side * 0.30
        end_lon = meet_lon + side * 0.50

        track = []
        for i in range(approach):
            f = i / approach
            track.append(
                (
                    start_lat + f * (meet_lat - start_lat),
                    start_lon + f * (meet_lon - start_lon),
                )
            )
        for _ in range(loiter):
            track.append(
                (
                    meet_lat + rng.gauss(0, 0.0006) + side * 0.0015,
                    meet_lon + rng.gauss(0, 0.0006) + side * 0.0015,
                )
            )
        for i in range(depart):
            f = (i + 1) / depart
            track.append(
                (
                    meet_lat + f * (end_lat - meet_lat),
                    meet_lon + f * (end_lon - meet_lon),
                )
            )
        tracks.append(track)
    return tracks


def _to_rows(mmsi: int, track: list[tuple[float, float]], rng: random.Random):
    """Turn a position track into AIS pings, deriving speed and course."""
    rows = []
    for i, (lat, lon) in enumerate(track):
        nxt = track[min(i + 1, len(track) - 1)]
        prv = track[max(i - 1, 0)]
        dlat = nxt[0] - prv[0]
        dlon = (nxt[1] - prv[1]) * math.cos(math.radians(lat))
        span = 2 if 0 < i < len(track) - 1 else 1

        dist_nm = math.hypot(dlat, dlon) * NM_PER_DEGREE
        hours = span * PING_SECONDS / 3600.0
        sog = dist_nm / hours if hours else 0.0
        cog = math.degrees(math.atan2(dlon, dlat)) % 360.0

        rows.append(
            {
                "mmsi": mmsi,
                "timestamp": START_TIME + timedelta(seconds=i * PING_SECONDS),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "sog": round(max(sog + rng.gauss(0, 0.15), 0.0), 2),
                "cog": round((cog + rng.gauss(0, 2.0)) % 360.0, 1),
            }
        )
    return rows


def build() -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    rows: list[dict] = []
    planted: dict[int, str] = {}
    mmsi = 200000000

    for _ in range(N_NORMAL):
        mmsi += 1
        rows += _to_rows(mmsi, _straight_track(rng), rng)

    # Anomaly 1: a vessel that goes dark for ~46 minutes.
    mmsi += 1
    gap_mmsi = mmsi
    planted[gap_mmsi] = "ais_gap"
    gap_rows = _to_rows(gap_mmsi, _straight_track(rng), rng)
    n = len(gap_rows)
    keep = [r for i, r in enumerate(gap_rows) if not (int(n * 0.45) <= i < int(n * 0.45) + 23)]
    rows += keep

    # Anomalies 2 and 3: a loitering rendezvous pair.
    for track in _rendezvous_tracks(rng):
        mmsi += 1
        planted[mmsi] = "rendezvous"
        rows += _to_rows(mmsi, track, rng)

    # Anomaly 4: erratic manoeuvring.
    mmsi += 1
    planted[mmsi] = "erratic"
    rows += _to_rows(mmsi, _erratic_track(rng), rng)

    df = pd.DataFrame(rows).sort_values(["timestamp", "mmsi"]).reset_index(drop=True)

    ensure_dirs()
    df.to_csv(RAW_AIS_CSV, index=False)
    with open(RAW_DIR / "planted_anomalies.txt", "w") as f:
        for m, kind in planted.items():
            f.write(f"{m},{kind}\n")

    return df


if __name__ == "__main__":
    df = build()
    print(f"wrote {len(df):,} pings for {df['mmsi'].nunique()} vessels -> {RAW_AIS_CSV}")
    print(f"time span: {df['timestamp'].min()}  ..  {df['timestamp'].max()}")
    print(f"sog range: {df['sog'].min():.2f} .. {df['sog'].max():.2f} knots")