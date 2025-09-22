# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from .io import read_yaml

def load_config(path: str) -> Dict[str, Any]:
    return read_yaml(path)

def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base
