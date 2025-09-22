# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Any, Dict, List, Generator

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            rows.append(json.loads(s))
    return rows

def stream_jsonl(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            yield json.loads(s)

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
