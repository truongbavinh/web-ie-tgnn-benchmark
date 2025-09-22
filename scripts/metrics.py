from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def accuracy(y_true, y_pred) -> Tuple[str, float]:
    return ("accuracy", float(accuracy_score(y_true, y_pred)))

def f1_macro(y_true, y_pred) -> Tuple[str, float]:
    return ("f1_macro", float(f1_score(y_true, y_pred, average="macro")))

METRIC_REGISTRY = {
    "accuracy": accuracy,
    "f1_macro": f1_macro,
}

def compute(metric_name: str, y_true, y_pred) -> Tuple[str, float]:
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {metric_name}")
    return METRIC_REGISTRY[metric_name](y_true, y_pred)
