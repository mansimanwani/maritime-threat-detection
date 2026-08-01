# maritime-threat-detection
# Maritime Threat Intelligence

Unsupervised anomaly detection over AIS vessel traffic using a graph neural
network, with per-anomaly explanations and a feedback loop that recalibrates
scoring from analyst verdicts.

## What it does

Ships broadcast AIS: identity, position, speed, heading, every couple of
minutes. This system slices that stream into time windows, links vessels that
were near each other into a spatio-temporal graph, and trains a graph
autoencoder to reconstruct normal traffic. Vessels it reconstructs badly get
flagged, each with a plain-language reason. An analyst confirms or dismisses
each flag, and those verdicts recalibrate how the system scores.

## Quickstart

Requires Python 3.13 and Node 20+.

```bash
# backend
python -m venv .venv
.venv\Scripts\activate                # macOS/Linux: source .venv/bin/activate
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

python run_pipeline.py                # builds everything, ~5s
uvicorn backend.api.main:app --reload --port 8000
```

```bash
# frontend, in a second terminal
cd frontend
npm install
npm run dev                           # http://localhost:5173
```

API docs at http://localhost:8000/docs.

## Pipeline

```
AIS pings -> clean -> window features -> spatio-temporal graph -> GNN autoencoder
          -> weighted score + threshold -> explanation -> SQLite -> FastAPI -> React/Leaflet
                                     ^                                             |
                                     +------------ recalibration <-- analyst verdict
```

| Stage | Module | Output |
|---|---|---|
| Generate AIS | `core/make_sample_data.py` | `data/raw/ais_sample.csv` |
| Clean | `core/ingest.py` | `data/processed/clean_pings.csv` |
| Features | `core/features.py` | `data/processed/features.csv` |
| Graph | `core/graph_builder.py` | `artifacts/graph.pt`, `scaler.json` |
| Train | `core/train.py` | `artifacts/gnn_autoencoder.pt` |
| Score | `core/detect.py` | `data/processed/scores.csv` |
| Explain | `core/explain.py` | `data/processed/anomalies.csv` |
| Store | `core/db.py` | `artifacts/app.db` |
| Recalibrate | `core/recalibrate.py` | `artifacts/calibration.json` |

## Design

**Graph.** 480 nodes (40 vessels x 12 half-hour windows). Spatial edges link
each vessel to its 5 nearest in the same window; temporal edges link each
vessel to itself in the next window. Both together make it spatio-temporal
rather than a stack of independent snapshots.

**Model.** GCN autoencoder, 5 -> 16 -> 3 -> 16 -> 5, 296 parameters. The
3-dimensional bottleneck is the mechanism: the network cannot memorise 480
nodes in 3 numbers, so it learns the majority pattern and fails on the rest.
Trained on all nodes -- no labels exist -- for 100 epochs. Longer training
measurably *degrades* detection, since the model eventually learns to
reconstruct the anomalies too.

**Explanations.** The score is a weighted sum of five per-feature
reconstruction errors, so each feature's share of that sum is its exact
contribution. Not a surrogate model, not an approximation -- the shares sum to
100% by construction. Wording is chosen against fleet percentiles, so a value
that is ordinary fleet-wide but wrong for its neighbourhood is described as
such rather than overstated.

**Recalibration.** Analyst verdicts do not retrain the network. They move two
knobs in `calibration.json`:

- `feature_weights`, nudged by (mean share on confirmed hits - mean share on
  false alarms), multiplicatively, then renormalised to mean 1.0
- `threshold_percentile`, a proportional controller on precision against a 90%
  target

Both updates are damped by `n / (n + 5)`, so a few verdicts nudge and many
move it properly. Feedback keys on `(window_id, mmsi)` rather than anomaly id,
because anomaly rows are regenerated on every recalibration.

## Results

Validated against planted anomalies (an AIS blackout, a loitering rendezvous
pair, an erratic manoeuvrer) whose identities the pipeline never reads.

Detection at default calibration: top 10 scored nodes are all genuinely
planted. Mean reconstruction error 0.35 for normal traffic versus 3.3-6.2 for
the planted vessels.

Recalibration over four rounds of full-list review:

| Version | Flagged | True | Precision |
|---|---|---|---|
| v0 | 24 | 19 | 79% |
| v1 | 22 | 17 | 77% |
| v2 | 20 | 17 | 85% |
| v3 | 19 | 17 | 89% |
| v4 | 19 | 18 | 95% |

Reproduce with `python simulate_feedback.py && python -m backend.core.recalibrate`,
repeated.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | calibration version, thresholds, counts |
| GET | `/api/anomalies` | flagged list with reasons, drivers, verdicts |
| GET | `/api/track/{mmsi}` | full vessel path for the map |
| POST | `/api/feedback` | record a verdict |
| POST | `/api/recalibrate` | close the loop, re-score, return the delta |

## Limitations

- **Data is synthetic.** Generated in the real AIS schema with planted
  anomalies so the detector can be validated. `ingest.py` accepts any CSV with
  `mmsi, timestamp, lat, lon, sog, cog` -- point `RAW_AIS_CSV` at a real feed
  (NOAA Marine Cadastre, Danish Maritime Authority) to swap it.
- **One region, six hours, 40 vessels.** Nothing here has been tested at the
  scale a real feed would produce.
- **Five features.** No vessel type, flag, cargo, draught, or port-call
  history, all of which a real system would use.
- **The feedback loop was validated against planted ground truth**, not real
  analysts. It demonstrates that the mechanism converges; it does not
  demonstrate that it converges on what a human expert would want.
- **No authentication.** Local development only.

## Stack

Python 3.13, PyTorch 2.12, PyTorch Geometric 2.7+, pandas, scikit-learn,
FastAPI, SQLite, React 19, Vite, Leaflet 1.9 / react-leaflet 5.