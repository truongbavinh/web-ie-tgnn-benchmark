# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Any
import re, unicodedata

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")
def finalize(attrs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(attrs)
    if isinstance(out.get("name"), str): out["name"] = _nfkc(out["name"]).strip()
    if isinstance(out.get("location"), str): out["location"] = _nfkc(out["location"]).strip()
    if isinstance(out.get("rating"), str):
        import re
        m = re.search(r'[0-9]+(?:\.[0-9]+)?', out["rating"])
        out["rating"] = float(m.group(0)) if m else None
    # amenities: dedup
    if isinstance(out.get("amenities"), list):
        seen=set(); arr=[]
        for a in out["amenities"]:
            if isinstance(a,str):
                t=_nfkc(a).strip()
                if t and t not in seen: seen.add(t); arr.append(t)
        out["amenities"] = arr
    # price: leave string (exporters handles money later or keep raw)
    return {k:v for k,v in out.items() if v not in (None,"",[],{})}
