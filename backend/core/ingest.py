"""Load raw AIS pings and clean them into a trustworthy table.

Real AIS is noisy: receivers drop bits, transponders emit placeholder
values, and the same ping is often picked up by two shore stations. Every
filter below removes one specific, well-known kind of junk.
"""

from __future__ import annotations

import pandas as pd

from backend.config import (
    RAW_AIS_CSV,
    CLEAN_PINGS_CSV,
    MIN_PINGS_PER_VESSEL,
    ensure_dirs,
)

REQUIRED_COLUMNS = ["mmsi", "timestamp", "lat", "lon", "sog", "cog"]

MAX_PLAUSIBLE_SOG = 60.0


def load_raw(path=RAW_AIS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No raw AIS file at {path}. Run: python -m backend.core.make_sample_data"
        )
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw AIS file is missing required columns: {missing}")

    return df[REQUIRED_COLUMNS].copy()


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    report: list[tuple[str, int]] = []

    def drop(mask: pd.Series, label: str) -> None:
        nonlocal df
        n = int(mask.sum())
        if n:
            df = df.loc[~mask].copy()
        report.append((label, n))

    start = len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["mmsi", "lat", "lon", "sog", "cog"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    drop(df[REQUIRED_COLUMNS].isna().any(axis=1), "unparseable or missing values")
    drop(~df["lat"].between(-90, 90), "latitude out of range")
    drop(~df["lon"].between(-180, 180), "longitude out of range")
    drop((df["lat"] == 0) & (df["lon"] == 0), "null island (0,0) placeholder")
    drop(~df["sog"].between(0, MAX_PLAUSIBLE_SOG), "implausible speed")
    drop(~df["cog"].between(0, 360, inclusive="left"), "course out of range")

    dupes = df.duplicated(subset=["mmsi", "timestamp"], keep="first")
    drop(dupes, "duplicate ping (same vessel, same time)")

    counts = df.groupby("mmsi")["timestamp"].transform("size")
    drop(counts < MIN_PINGS_PER_VESSEL, f"vessel had under {MIN_PINGS_PER_VESSEL} pings")

    df["mmsi"] = df["mmsi"].astype("int64")
    df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    if verbose:
        print(f"raw rows: {start:,}")
        for label, n in report:
            print(f"  dropped {n:>6,}  {label}")
        print(f"clean rows: {len(df):,}  ({df['mmsi'].nunique()} vessels)")

    return df


def main() -> pd.DataFrame:
    ensure_dirs()
    df = clean(load_raw())
    df.to_csv(CLEAN_PINGS_CSV, index=False)
    print(f"wrote -> {CLEAN_PINGS_CSV}")
    return df


if __name__ == "__main__":
    main()