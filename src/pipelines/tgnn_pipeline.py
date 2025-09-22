# -*- coding: utf-8 -*-
"""
TGNN pipeline (topology-aware GNN) for IE.
Interfaces:
    train(graph_path: str, encoder_ckpt: str, class_weights_path: Optional[str],
          ckpt_out: str, seed: int = 42)

    predict(graph_path: str, model_ckpt: str, domain: str, out_path: str)

Assumptions:
- graph_path: a saved graph dataset (PyTorch Geometric InMemoryDataset or a list of Data objects),
             produced by src.data.graph_builder.build_graph_for_domain(...)
  Each Data has:
      .x         (node features)           [N, D]
      .edge_index (graph edges)            [2, E]
      .y         (node labels or edge labels) [N] or [E] depending on your task
      .meta      (optional: ids to map back to tokens/spans)
- encoder_ckpt: path to BERT checkpoint to init encoder (optional usage: features frozen or as text encoder)
- class_weights_path: JSON mapping {label_id: weight} for imbalanced loss (optional)
- predict(): writes a RAW per-node/per-span result to `out_path` (JSONL) that exporter will convert
            into `attributes` per schema.md.
"""

import os, json, random
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# If you use PyTorch Geometric
try:
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, SAGEConv
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False

# ----------------------------
# Model definition (example)
# ----------------------------
class SimpleTGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, conv: str = "sage", num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        Conv = SAGEConv if conv == "sage" else GCNConv
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        for i in range(num_layers):
            self.convs.append(Conv(dims[i], dims[i+1] if i+1 < len(dims) else hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        logits = self.head(x)   # node classification
        return logits

# ----------------------------
# Helpers
# ----------------------------
def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _load_graphs(graph_path: str):
    """
    Expected to return an iterable of PyG Data objects (or a Dataset).
    Implement this according to how you saved graphs in graph_builder.py
    """
    # Example: torch.load returns a list of Data
    obj = torch.load(graph_path, map_location="cpu")
    return obj  # List[Data] or Dataset

def _load_class_weights(path: Optional[str], num_classes: int):
    if not path: return None
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    w = torch.ones(num_classes, dtype=torch.float)
    for k,v in m.items():
        k = int(k)
        if 0 <= k < num_classes:
            w[k] = float(v)
    return w

# ----------------------------
# Public API
# ----------------------------
def train(graph_path: str, encoder_ckpt: str, class_weights_path: Optional[str], ckpt_out: str, seed: int = 42):
    assert _HAS_PYG, "torch_geometric is required for this TGNN example."
    os.makedirs(ckpt_out, exist_ok=True)
    _set_seed(seed)

    # 1) Load graphs
    dataset = _load_graphs(graph_path)
    if isinstance(dataset, list):
        # naive split: 80/20
        n = len(dataset)
        cut = int(0.8*n)
        train_set, val_set = dataset[:cut], dataset[cut:]
    else:
        # If it's a Dataset, split yourself or use masks
        train_set = dataset
        val_set = dataset[:0]

    # 2) Infer dims (adapt to your Data fields)
    sample = train_set[0]
    in_dim = sample.x.size(-1)
    num_classes = int(sample.y.max().item()) + 1

    # 3) Build model
    model = SimpleTGNN(in_dim=in_dim, hidden_dim=int(os.environ.get("TGNN_HID", 128)),
                       out_dim=num_classes, conv=os.environ.get("TGNN_CONV","sage"),
                       num_layers=int(os.environ.get("TGNN_LAYERS", 2)),
                       dropout=float(os.environ.get("TGNN_DROPOUT", 0.2)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) Loss & optimizer
    class_weights = _load_class_weights(class_weights_path, num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get("TGNN_LR", 3e-4)))

    train_loader = DataLoader(train_set, batch_size=int(os.environ.get("TGNN_BS", 1)), shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False)

    best_val = None
    epochs = int(os.environ.get("TGNN_EPOCHS", 10))
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))

        # simple val acc
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                preds = logits.argmax(-1)
                correct += int((preds == batch.y).sum().item())
                total += int(batch.y.numel())
        val_acc = (correct / total) if total else 0.0
        print(f"[Epoch {ep}] loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if best_val is None or val_acc >= best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(),
                        "in_dim": in_dim,
                        "num_classes": num_classes,
                        "hparams": {"hidden": int(os.environ.get("TGNN_HID", 128))}},
                       os.path.join(ckpt_out, "tgnn.ckpt"))
    # Done
    print(f"Saved best TGNN to {os.path.join(ckpt_out,'tgnn.ckpt')}")

def predict(graph_path: str, model_ckpt: str, domain: str, out_path: str):
    assert _HAS_PYG, "torch_geometric is required for this TGNN example."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load model
    ckpt = torch.load(model_ckpt, map_location="cpu")
    in_dim = ckpt["in_dim"]; num_classes = ckpt["num_classes"]
    model = SimpleTGNN(in_dim=in_dim, hidden_dim=ckpt["hparams"]["hidden"], out_dim=num_classes)
    model.load_state_dict(ckpt["model"]); model.to(device); model.eval()

    # 2) Load graphs
    dataset = _load_graphs(graph_path)
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 3) Write RAW predictions (one JSON per sample)
    #    RAW format example (flexible): {"id": "...", "domain": "...", "nodes": [{"idx": i, "label": 3, "score": 0.91, "span": [s,e]}], "edges": ...}
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                logits = model(batch.x, batch.edge_index)
                probs = torch.softmax(logits, dim=-1)
                pred = probs.argmax(-1)
            # You likely stored mapping info in batch (e.g., batch.meta) to map back to tokens/spans
            # Here we emit per-node predictions; exporter will map these to attributes per domain
            record = {
                "id": getattr(batch, "doc_id", ["unknown"])[0] if hasattr(batch, "doc_id") else "unknown",
                "domain": domain,
                "nodes": [
                    {"idx": int(i), "label": int(lbl), "score": float(probs[i, lbl])}
                    for i, lbl in enumerate(pred.cpu().tolist())
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote RAW predictions to {out_path}")
