"""SQLite storage for anomalies and analyst feedback.

Two tables, and one design decision worth stating up front:

  anomalies  is REPLACED on every detection run. It is a snapshot of what
             the current calibration flags -- not a permanent record.
  feedback   is APPEND-ONLY and keys on (window_id, mmsi), never on an
             anomaly row id. Anomaly rows get regenerated after every
             recalibration; a verdict about a vessel-window does not.

If feedback pointed at anomaly ids, the first recalibration would orphan
every verdict an analyst had given, and the loop could never close.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from backend.config import ANOMALIES_CSV, DB_PATH, ensure_dirs

VALID_LABELS = ("true_positive", "false_positive")

SCHEMA = """
CREATE TABLE IF NOT EXISTS anomalies (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id           INTEGER NOT NULL,
    window_start        TEXT    NOT NULL,
    mmsi                INTEGER NOT NULL,
    lat                 REAL    NOT NULL,
    lon                 REAL    NOT NULL,
    score               REAL    NOT NULL,
    reason              TEXT    NOT NULL,
    drivers             TEXT    NOT NULL,
    calibration_version INTEGER NOT NULL,
    UNIQUE (window_id, mmsi)
);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id  INTEGER NOT NULL,
    mmsi       INTEGER NOT NULL,
    label      TEXT    NOT NULL CHECK (label IN ('true_positive', 'false_positive')),
    drivers    TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_target ON feedback (window_id, mmsi);
"""


def get_connection() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def replace_anomalies(anomalies: pd.DataFrame, calibration_version: int) -> int:
    """Wipe the snapshot and write the current one."""
    rows = [
        (
            int(r.window_id),
            str(r.window_start),
            int(r.mmsi),
            float(r.lat),
            float(r.lon),
            float(r.score),
            str(r.reason),
            str(r.drivers),
            int(calibration_version),
        )
        for r in anomalies.itertuples()
    ]
    with get_connection() as conn:
        conn.execute("DELETE FROM anomalies")
        conn.executemany(
            """INSERT INTO anomalies
               (window_id, window_start, mmsi, lat, lon, score, reason, drivers, calibration_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return len(rows)


def fetch_anomalies(limit: int | None = None) -> list[dict]:
    """Current anomalies, each carrying the analyst's latest verdict if any."""
    sql = """
        SELECT a.*, f.label AS feedback_label
        FROM anomalies AS a
        LEFT JOIN (
            SELECT window_id, mmsi, label
            FROM feedback
            WHERE id IN (SELECT MAX(id) FROM feedback GROUP BY window_id, mmsi)
        ) AS f
          ON f.window_id = a.window_id AND f.mmsi = a.mmsi
        ORDER BY a.score DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"

    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()

    out = []
    for row in rows:
        item = dict(row)
        item["drivers"] = json.loads(item["drivers"])
        out.append(item)
    return out


def insert_feedback(window_id: int, mmsi: int, label: str, drivers: dict) -> int:
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")

    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO feedback (window_id, mmsi, label, drivers, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                int(window_id),
                int(mmsi),
                label,
                json.dumps(drivers),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        return int(cur.lastrowid)


def fetch_feedback() -> pd.DataFrame:
    """Latest verdict per (window_id, mmsi), with drivers parsed."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT window_id, mmsi, label, drivers, created_at
               FROM feedback
               WHERE id IN (SELECT MAX(id) FROM feedback GROUP BY window_id, mmsi)"""
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["window_id", "mmsi", "label", "drivers", "created_at"])

    df = pd.DataFrame([dict(r) for r in rows])
    df["drivers"] = df["drivers"].map(json.loads)
    return df


def clear_feedback() -> int:
    """Reset the loop so the demo can be run again from scratch."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM feedback")
        return cur.rowcount


def main() -> None:
    from backend.core.detect import load_calibration

    if not ANOMALIES_CSV.exists():
        raise FileNotFoundError(
            f"No anomalies at {ANOMALIES_CSV}. Run: python -m backend.core.explain"
        )

    init_db()
    version = load_calibration()["version"]
    n = replace_anomalies(pd.read_csv(ANOMALIES_CSV), version)

    print(f"database        {DB_PATH}")
    print(f"anomalies       {n} rows written (calibration version {version})")
    print(f"feedback        {len(fetch_feedback())} verdict(s) on record\n")

    for item in fetch_anomalies(limit=3):
        print(f"  #{item['id']} window {item['window_id']} vessel {item['mmsi']}  score {item['score']:.2f}")
        print(f"     verdict: {item['feedback_label'] or '-- none yet --'}")
        top = max(item["drivers"], key=item["drivers"].get)
        print(f"     top driver: {top} ({item['drivers'][top]:.0%})")


if __name__ == "__main__":
    main()