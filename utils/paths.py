# -*- coding: utf-8 -*-
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
