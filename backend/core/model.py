"""A graph autoencoder for unsupervised anomaly detection.

The model squeezes each node's five features -- mixed with its neighbours'
features -- down to a tiny embedding, then tries to rebuild the original
five numbers from it. Nodes it rebuilds badly are the ones that don't fit
the pattern the rest of the traffic follows.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from backend.config import EMBED_DIM, HIDDEN_DIM


class GNNAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.enc1 = GCNConv(in_dim, hidden_dim)
        self.enc2 = GCNConv(hidden_dim, embed_dim)
        self.dec1 = GCNConv(embed_dim, hidden_dim)
        self.dec2 = GCNConv(hidden_dim, in_dim)

    def encode(self, x, edge_index):
        h = torch.relu(self.enc1(x, edge_index))
        return self.enc2(h, edge_index)

    def decode(self, z, edge_index):
        h = torch.relu(self.dec1(z, edge_index))
        return self.dec2(h, edge_index)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index), z


def reconstruction_error(x, x_hat):
    """Squared error per node per feature. Keeping the per-feature detail is
    what makes the explainability layer possible later."""
    return (x - x_hat) ** 2