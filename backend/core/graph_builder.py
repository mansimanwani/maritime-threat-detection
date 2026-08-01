"""Build one spatio-temporal graph from the windowed feature table.

Nodes  : one per (vessel, time window)  -- 40 vessels x 12 windows = 480
Spatial edges  : within a window, each vessel links to its GRAPH_KNN nearest vessels
Temporal edges : the same vessel in consecutive windows links to itself

The result is a single PyTorch Geometric `Data` object, so the GNN can
pass messages both sideways (what were my neighbours doing right now?)
and forwards (what was I doing half an hour ago?).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from backend.config import (
    FEATURES_CSV,
    FEATURE_COLUMNS,
    GRAPH_KNN,
    GRAPH_PATH,
    SCALER_PATH,
    ensure_dirs,
)
from backend.core.features import haversine_km


def _spatial_edges(window: pd.DataFrame) -> list[tuple[int, int]]:
    """Link each vessel to its GRAPH_KNN nearest vessels in the same window."""
    idx = window.index.to_numpy()
    lat = window["lat"].to_numpy()
    lon = window["lon"].to_numpy()

    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(dist, np.inf)

    k = min(GRAPH_KNN, len(idx) - 1)
    edges = []
    for i, nearest in enumerate(np.argsort(dist, axis=1)[:, :k]):
        for j in nearest:
            edges.append((int(idx[i]), int(idx[j])))
    return edges


def _temporal_edges(nodes: pd.DataFrame) -> list[tuple[int, int]]:
    """Link each vessel to itself in the next window."""
    edges = []
    for _, track in nodes.groupby("mmsi", sort=False):
        track = track.sort_values("window_id")
        rows = track.index.to_numpy()
        for a, b in zip(rows[:-1], rows[1:]):
            edges.append((int(a), int(b)))
    return edges


def _to_undirected(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Both directions, no duplicates."""
    both = {(a, b) for a, b in edges} | {(b, a) for a, b in edges}
    pairs = sorted(both)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def build_graph(nodes: pd.DataFrame) -> tuple[Data, StandardScaler]:
    nodes = nodes.sort_values(["window_id", "mmsi"]).reset_index(drop=True)

    # One scaler fitted across every window, not per window: otherwise a
    # quiet window would have its tiny variations stretched into "anomalies".
    scaler = StandardScaler()
    x = scaler.fit_transform(nodes[FEATURE_COLUMNS].to_numpy(dtype=np.float64))

    spatial, temporal = [], []
    for _, window in nodes.groupby("window_id", sort=True):
        spatial += _spatial_edges(window)
    temporal += _temporal_edges(nodes)

    edge_index = _to_undirected(spatial + temporal)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        mmsi=torch.tensor(nodes["mmsi"].to_numpy(), dtype=torch.long),
        window_id=torch.tensor(nodes["window_id"].to_numpy(), dtype=torch.long),
    )
    data.n_spatial = len(set(spatial))
    data.n_temporal = len(set(temporal))
    return data, scaler


def save_graph(data: Data) -> None:
    """Save as plain tensors.

    We deliberately do not pickle the Data object: since PyTorch 2.6
    `torch.load` defaults to weights_only=True and refuses pickled classes.
    Saving a dict of tensors keeps that safety default intact.
    """
    torch.save(
        {
            "x": data.x,
            "edge_index": data.edge_index,
            "mmsi": data.mmsi,
            "window_id": data.window_id,
        },
        GRAPH_PATH,
    )


def load_graph() -> Data:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"No graph at {GRAPH_PATH}. Run: python -m backend.core.graph_builder"
        )
    blob = torch.load(GRAPH_PATH)
    return Data(
        x=blob["x"],
        edge_index=blob["edge_index"],
        mmsi=blob["mmsi"],
        window_id=blob["window_id"],
    )


def main() -> Data:
    ensure_dirs()
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"No features at {FEATURES_CSV}. Run: python -m backend.core.features"
        )

    nodes = pd.read_csv(FEATURES_CSV)
    data, scaler = build_graph(nodes)

    save_graph(data)
    with open(SCALER_PATH, "w") as f:
        json.dump(
            {
                "features": FEATURE_COLUMNS,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            f,
            indent=2,
        )

    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
    print(f"nodes            {data.num_nodes}  ({nodes['mmsi'].nunique()} vessels x {nodes['window_id'].nunique()} windows)")
    print(f"node features    {data.num_node_features}  {FEATURE_COLUMNS}")
    print(f"directed edges   {data.num_edges}")
    print(f"  spatial pairs  {data.n_spatial}")
    print(f"  temporal pairs {data.n_temporal}")
    print(f"degree           min {int(deg.min())}, mean {deg.float().mean():.1f}, max {int(deg.max())}")
    print(f"isolated nodes   {int((deg == 0).sum())}")
    print(f"\nwrote -> {GRAPH_PATH}")
    print(f"wrote -> {SCALER_PATH}")
    return data


if __name__ == "__main__":
    main()