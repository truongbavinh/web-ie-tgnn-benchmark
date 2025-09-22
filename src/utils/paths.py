# -*- coding: utf-8 -*-
"""Project path helpers to keep repo layout consistent."""
from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    # assume this file is under <root>/src/utils/paths.py
    return Path(__file__).resolve().parents[3]

def repo_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)

def data_path(*parts: str) -> Path:
    return repo_path("data", *parts)

def results_path(*parts: str) -> Path:
    return repo_path("results", *parts)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
