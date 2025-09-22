# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple, Set
import unicodedata

PRICE_LIKE = {"price", "fees", "amount"}

def nfkc_lower_trim(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return s

def norm_scalar(attr: str, v: Any) -> Any:
    if v is None: return None
    if isinstance(v, str): return nfkc_lower_trim(v)
    if isinstance(v, (int, float)): return v
    return v

def norm_list(xs: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen: Set[Any] = set()
    for x in xs:
        nx = norm_scalar("_", x)
        if nx in seen: continue
        seen.add(nx); out.append(nx)
    return out

def is_price_obj(v: Any) -> bool:
    return isinstance(v, dict) and "value" in v and "currency" in v

def is_area_obj(v: Any) -> bool:
    return isinstance(v, dict) and "value" in v and "unit" in v

def norm_price(v: Dict[str, Any]) -> Tuple[str, float]:
    cur = nfkc_lower_trim(str(v.get("currency","")))
    try: val = float(v.get("value", 0.0))
    except Exception: val = 0.0
    return (cur, round(val, 2))

def norm_area(v: Dict[str, Any]) -> Tuple[str, float]:
    unit = nfkc_lower_trim(str(v.get("unit","")))
    try: val = float(v.get("value", 0.0))
    except Exception: val = 0.0
    return (unit, val)

def instances_from_attributes(attrs: Dict[str, Any]):
    inst = set()
    for attr, v in attrs.items():
        if v is None: continue
        if attr in PRICE_LIKE and is_price_obj(v):
            cur, val = norm_price(v)
            inst.add((attr, ("money", cur, val))); continue
        if is_area_obj(v):
            unit, val = norm_area(v)
            inst.add((attr, ("area", unit, val))); continue
        if isinstance(v, list):
            for item in norm_list(v):
                inst.add((attr, item))
            continue
        inst.add((attr, norm_scalar(attr, v)))
    return inst

def equal_attr_value(attr: str, gold_v: Any, pred_v: Any) -> bool:
    if attr in PRICE_LIKE and is_price_obj(gold_v) and is_price_obj(pred_v):
        gcur, gval = norm_price(gold_v); pcur, pval = norm_price(pred_v)
        if gcur != pcur: return False
        return abs(gval - pval) <= 0.01
    if is_area_obj(gold_v) and is_area_obj(pred_v):
        gunit, gval = norm_area(gold_v); punit, pval = norm_area(pred_v)
        return (gunit == punit) and (gval == pval)
    if isinstance(gold_v, list) and isinstance(pred_v, list):
        return set(norm_list(gold_v)) == set(norm_list(pred_v))
    return norm_scalar(attr, gold_v) == norm_scalar(attr, pred_v)
