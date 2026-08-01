"""Train the graph autoencoder on the whole graph.

There are no labels, so we train on everything. That works because odd
behaviour is rare: the model spends its limited capacity learning the
majority pattern, and ends up unable to reproduce the few nodes that
break it. Those leftovers are our anomalies.
"""

from __future__ import annotations

import torch

from backend.config import EPOCHS, LEARNING_RATE, MODEL_PATH, RANDOM_SEED, ensure_dirs
from backend.core.graph_builder import load_graph
from backend.core.model import GNNAutoencoder, reconstruction_error


def train(data, epochs: int = EPOCHS, lr: float = LEARNING_RATE, verbose: bool = True):
    torch.manual_seed(RANDOM_SEED)

    model = GNNAutoencoder(in_dim=data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_hat, _ = model(data.x, data.edge_index)
        loss = reconstruction_error(data.x, x_hat).mean()
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:>4}   loss {loss.item():.4f}")

    return model


def main():
    ensure_dirs()
    data = load_graph()

    print(f"training on {data.num_nodes} nodes, {data.num_edges} edges")
    model = train(data)

    n_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), MODEL_PATH)

    model.eval()
    with torch.no_grad():
        x_hat, z = model(data.x, data.edge_index)
        per_node = reconstruction_error(data.x, x_hat).mean(dim=1)

    print(f"\nparameters      {n_params}")
    print(f"embedding shape {tuple(z.shape)}")
    print(f"node error      min {per_node.min():.3f}  median {per_node.median():.3f}  max {per_node.max():.3f}")
    print(f"\nwrote -> {MODEL_PATH}")
    return model


if __name__ == "__main__":
    main()