import sys
import torch
import torch_geometric
import pandas as pd
import numpy as np
import sklearn
import fastapi
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__)
print("torch_geom  ", torch_geometric.__version__)
print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("scikit-learn", sklearn.__version__)
print("fastapi     ", fastapi.__version__)

# Smoke test: 3 nodes, 2 edges, 2 features each -> one GCN layer
x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

out = GCNConv(2, 4)(data.x, data.edge_index)
print("GCNConv output shape:", tuple(out.shape), "(expected (3, 4))")
print("ENVIRONMENT OK")