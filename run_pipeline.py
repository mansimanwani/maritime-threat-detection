"""Rebuild everything from raw AIS in one command.

    python run_pipeline.py              # full rebuild, keeps any feedback
    python run_pipeline.py --reset      # also wipes feedback and calibration
    python run_pipeline.py --keep-data  # reuse the existing raw AIS file

Each stage is one function you can also run on its own with
`python -m backend.core.<stage>`; this just runs them in order.
"""

from __future__ import annotations

import argparse
import time

from backend.config import CALIBRATION_PATH, RAW_AIS_CSV, ensure_dirs


# (label, module, entry function)
STAGES = [
    ("generate raw AIS", "backend.core.make_sample_data", "build"),
    ("clean pings", "backend.core.ingest", "main"),
    ("build features", "backend.core.features", "main"),
    ("build graph", "backend.core.graph_builder", "main"),
    ("train GNN", "backend.core.train", "main"),
    ("score and threshold", "backend.core.detect", "main"),
    ("explain anomalies", "backend.core.explain", "main"),
    ("load database", "backend.core.db", "main"),
]


def run(skip_data: bool = False) -> None:
    import importlib

    started = time.time()
    for i, (label, module_path, entry) in enumerate(STAGES, start=1):
        if skip_data and module_path.endswith("make_sample_data"):
            print(f"[{i}/{len(STAGES)}] {label} ... skipped (reusing {RAW_AIS_CSV.name})")
            continue

        print(f"[{i}/{len(STAGES)}] {label} ...", flush=True)
        module = importlib.import_module(module_path)
        getattr(module, entry)()
        print()

    print(f"pipeline complete in {time.time() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the maritime threat intelligence pipeline.")
    parser.add_argument("--reset", action="store_true", help="wipe feedback and calibration first")
    parser.add_argument("--keep-data", action="store_true", help="reuse the existing raw AIS file")
    args = parser.parse_args()

    ensure_dirs()

    if args.reset:
        from backend.core.db import clear_feedback, init_db

        init_db()
        n = clear_feedback()
        CALIBRATION_PATH.unlink(missing_ok=True)
        print(f"reset: {n} verdict(s) deleted, calibration back to defaults\n")

    if args.keep_data and not RAW_AIS_CSV.exists():
        raise SystemExit(f"--keep-data given but {RAW_AIS_CSV} does not exist")

    run(skip_data=args.keep_data)


if __name__ == "__main__":
    main()