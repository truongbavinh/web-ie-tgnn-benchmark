# -*- coding: utf-8 -*-
"""
TGNN model identical to the one defined/used in training_tgnn.py.

Layers & params (mirrored):
- GCNConv with add_self_loops=False
- BatchNorm after each GCN layer
- ReLU activation
- Dropout between GCN blocks (not after the last one)
- Final Linear(hidden_dim -> num_classes)

Forward expects a torch_geometric.data.Data with:
  .x           [N, bert_hidden_size]
  .edge_index  [2, E]
  .y           [N]   (for training; not used here)

This file is state_dict-compatible with the training script.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class TGNN_BERT(nn.Module):
    def __init__(
        self,
        bert_hidden_size: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_gcn_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        assert num_gcn_layers >= 1, "num_gcn_layers must be >= 1"

        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First GCN layer: in_dim=bert_hidden_size -> hidden_dim
        self.gcn_layers.append(GCNConv(bert_hidden_size, hidden_dim, add_self_loops=False))
        self.batch_norms.append(BatchNorm(hidden_dim))

        # Additional hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = data.x.to(torch.float32)
        edge_index = data.edge_index
        # Pass through GCN + BN + ReLU (+ Dropout except last)
        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.gcn_layers) - 1:
                x = self.dropout(x)
        logits = self.classifier(x)
        return logits
