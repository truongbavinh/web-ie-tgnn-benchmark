# -*- coding: utf-8 -*-
"""Samplers & DataLoaders for TGNN graphs (PyTorch Geometric)."""
from typing import List, Tuple, Optional
import os, glob, random, torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_graph_dir(graph_dir: str) -> List[Data]:
    files = sorted(glob.glob(os.path.join(graph_dir, "*.pt")))
    dataset: List[Data] = []
    for fp in files:
        try:
            d = torch.load(fp, map_location="cpu")
            if isinstance(d, list):
                dataset.extend(d)
            else:
                dataset.append(d)
        except Exception as e:
            print(f"⚠️ Skip {fp}: {e}")
    if not dataset:
        raise FileNotFoundError(f"No .pt graphs found in {graph_dir}")
    return dataset

def train_val_split(dataset: List[Data], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Data], List[Data]]:
    random.Random(seed).shuffle(dataset)
    n = len(dataset)
    cut = int(round(n * (1 - val_ratio)))
    train_set = dataset[:cut] if cut > 0 else []
    val_set   = dataset[cut:] if cut < n else []
    return train_set, val_set

def make_loaders(train_set: List[Data], val_set: Optional[List[Data]] = None,
                 batch_size: int = 1, num_workers: int = 0, shuffle_train: bool = True):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_loader = None
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
