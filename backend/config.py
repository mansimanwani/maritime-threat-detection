"""Single source of truth for paths and tunable constants."""

from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "backend" / "artifacts"

RAW_AIS_CSV = RAW_DIR / "ais_sample.csv"
CLEAN_PINGS_CSV = PROCESSED_DIR / "clean_pings.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
SCORES_CSV = PROCESSED_DIR / "scores.csv"
ANOMALIES_CSV = PROCESSED_DIR / "anomalies.csv"

MODEL_PATH = ARTIFACTS_DIR / "gnn_autoencoder.pt"
CALIBRATION_PATH = ARTIFACTS_DIR / "calibration.json"
DB_PATH = ARTIFACTS_DIR / "app.db"
GRAPH_PATH = ARTIFACTS_DIR / "graph.pt"
SCALER_PATH = ARTIFACTS_DIR / "scaler.json"

# ---------- Graph construction ----------
WINDOW_MINUTES = 30          # length of one time slice
PROXIMITY_KM = 10.0          # two vessels get an edge if closer than this
GRAPH_KNN = 5      # keeps dense port areas from exploding

# ---------- Features ----------
MIN_PINGS_PER_VESSEL = 10    # a vessel with fewer pings has no behaviour to model

FEATURE_COLUMNS = [
    "mean_speed",
    "speed_change",
    "course_change",
    "gap_minutes",
    "neighbor_count",
]

# ---------- Model ----------
HIDDEN_DIM = 16
EMBED_DIM = 3
EPOCHS = 100
LEARNING_RATE = 0.01
RANDOM_SEED = 42

# ---------- Detection / calibration defaults ----------
THRESHOLD_PERCENTILE = 95.0                     # flag the top 5% by default
DEFAULT_FEATURE_WEIGHTS = {f: 1.0 for f in FEATURE_COLUMNS}
EXPLAIN_TOP_K = 2          # name at most 2 driving features
EXPLAIN_MIN_SHARE = 0.20   # ...and only if the 2nd one carries 20%+ of the score


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

# ---------- Feedback recalibration ----------
RECALIBRATION_RATE = 0.8       # how hard evidence moves a weight
WEIGHT_MIN = 0.2               # a feature can be quietened, never silenced
WEIGHT_MAX = 3.0
FEEDBACK_PRIOR = 5             # damping: 5 verdicts move less than 50

TARGET_PRECISION = 0.90        # analyst time is expensive; aim high
THRESHOLD_STEP = 5.0           # percentile points per unit of precision error
THRESHOLD_BOUNDS = (80.0, 99.0)

CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]