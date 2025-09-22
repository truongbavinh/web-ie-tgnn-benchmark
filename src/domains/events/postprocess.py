# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Any
import re, unicodedata

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")
def finalize(a: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k in ["name","venue","date_time"]:
        if isinstance(out.get(k), str): out[k] = _nfkc(out[k]).strip()
    if isinstance(out.get("artists"), list):
        seen=set(); arr=[]
        for x in out["artists"]:
            if isinstance(x,str):
                t=_nfkc(x).strip()
                if t and t not in seen: seen.add(t); arr.append(t)
        out["artists"]=arr
    return {k:v for k,v in out.items() if v not in (None,"",[],{})}
