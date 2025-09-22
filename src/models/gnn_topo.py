# -*- coding: utf-8 -*-
"""Topology-aware GNN building blocks (PyTorch Geometric)."""
from typing import Literal
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm

class TGNNStack(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2,
                 conv: Literal["sage","gcn"] = "sage", dropout: float = 0.2):
        super().__init__()
        Conv = SAGEConv if conv == "sage" else GCNConv
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(Conv(in_dim, hidden_dim, add_self_loops=False if conv=="gcn" else True))
        self.bns.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(Conv(hidden_dim, hidden_dim, add_self_loops=False if conv=="gcn" else True))
            self.bns.append(BatchNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return x
