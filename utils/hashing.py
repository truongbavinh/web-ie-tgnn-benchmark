# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib
from pathlib import Path

def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha1sum(path: str | Path) -> str:
    return _hash_file(Path(path), "sha1")

def sha256sum(path: str | Path) -> str:
    return _hash_file(Path(path), "sha256")

def verify_hash(path: str | Path, expected: str) -> bool:
    p = Path(path)
    exp = expected.lower()
    got = sha1sum(p) if len(exp) == 40 else sha256sum(p)
    return got.lower() == exp
