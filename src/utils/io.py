# -*- coding: utf-8 -*-
"""Lightweight IO helpers: JSONL/YAML/CSV with UTF-8 and atomic writes."""
from __future__ import annotations
from typing import Iterable, Dict, Any, List, Optional
import json, csv, os, tempfile, shutil

try:
    import yaml  # optional
except Exception:
    yaml = None

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is not installed")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def write_yaml(path: str, data: Dict[str, Any]) -> None:
    if yaml is None:
        raise ImportError("PyYAML is not installed")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    rows = list(rows)
    if not fieldnames and rows:
        # infer from union of keys
        ks = set()
        for r in rows: ks.update(r.keys())
        fieldnames = list(ks)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or [])
        w.writeheader()
        for r in rows: w.writerow(r)

def atomic_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dname = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=dname, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        shutil.move(tmp, path)
    finally:
        try: os.remove(tmp)
        except FileNotFoundError: pass
