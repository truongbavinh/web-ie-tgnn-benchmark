# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Any
import re, unicodedata

_SIZE_CANON = ["XXS","XS","S","M","L","XL","XXL","2XL","3XL","4XL"]
_COLOR_WORDS = {"black","white","red","green","blue","yellow","pink","purple","brown","grey","gray","beige","navy","khaki","ivory","cream","ecru","gold","silver","orange","burgundy"}
_MAT_MAP = {"cotton":"cotton","linen":"linen","leather":"leather","polyester":"polyester","wool":"wool","silk":"silk","cashmere":"cashmere","nylon":"nylon","acrylic":"acrylic","viscose":"viscose","elastane":"elastane","spandex":"elastane","rayon":"rayon"}

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def canonicalize_sizes(sizes: List[str]) -> List[str]:
    out = []
    for s in sizes:
        t = _nfkc(s).upper().strip().replace(" ", "")
        t = t.replace("EXTRA SMALL","XS").replace("EXTRASMALL","XS").replace("SMALL","S").replace("MEDIUM","M").replace("LARGE","L").replace("EXTRA LARGE","XL").replace("EXTRALARGE","XL")
        if t in _SIZE_CANON or re.fullmatch(r"[0-9]{2}", t):
            if t not in out: out.append(t)
    return out or sizes

def canonicalize_materials(mats: List[str]) -> List[str]:
    out = []
    for m in mats:
        t = _nfkc(m).lower()
        t = re.sub(r"[^a-z ]+", " ", t).strip()
        keep = []
        for k,v in _MAT_MAP.items():
            if re.search(rf"\b{k}\b", t):
                keep.append(v)
        if keep:
            for k in keep:
                if k not in out: out.append(k)
        else:
            short = " ".join(t.split()[:3])
            if short and short not in out:
                out.append(short)
    return out

def canonicalize_color(c: str) -> str:
    t = _nfkc(c).lower().strip()
    for w in _COLOR_WORDS:
        if re.search(rf"\b{w}\b", t):
            return w
    return t

def finalize(attributes: Dict[str, Any]) -> Dict[str, Any]:
    attrs = dict(attributes)
    if "name" in attrs and isinstance(attrs["name"], str):
        attrs["name"] = _nfkc(attrs["name"]).strip()
    if "price" in attrs and isinstance(attrs["price"], dict):
        cur = attrs["price"].get("currency"); val = attrs["price"].get("value")
        try:
            if isinstance(cur, str): attrs["price"]["currency"] = cur.upper().strip()
        except Exception: pass
        try:
            attrs["price"]["value"] = float(val)
        except Exception:
            attrs.pop("price", None)
    if "material" in attrs and isinstance(attrs["material"], list):
        mats = canonicalize_materials([m for m in attrs["material"] if isinstance(m, str)])
        seen=set(); nm=[]
        for m in mats:
            if m not in seen: seen.add(m); nm.append(m)
        attrs["material"] = nm
    if "color" in attrs and isinstance(attrs["color"], str):
        attrs["color"] = canonicalize_color(attrs["color"])
    if "size" in attrs and isinstance(attrs["size"], list):
        sizes = canonicalize_sizes([s for s in attrs["size"] if isinstance(s, str)])
        seen=set(); ns=[]
        for s in sizes:
            if s not in seen: seen.add(s); ns.append(s)
        attrs["size"] = ns
    return {k:v for k,v in attrs.items() if v not in (None, "", [], {})}
