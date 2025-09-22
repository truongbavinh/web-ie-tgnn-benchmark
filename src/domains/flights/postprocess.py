# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Any
import re, unicodedata

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")
def finalize(a: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k in ["name","duration","departure_time","arrival_time","airline"]:
        if isinstance(out.get(k), str): out[k] = _nfkc(out[k]).strip()
    return {k:v for k,v in out.items() if v not in (None,"",[],{})}
