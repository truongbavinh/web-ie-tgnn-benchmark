# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
from .utils import PRICE_LIKE, is_price_obj, norm_price

def price_mae(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> Dict[str, float]:
    pred_map = {r["id"]: r for r in pred}
    diffs = []
    for g in gold:
        p = pred_map.get(g["id"], {"attributes": {}})
        ga = g.get("attributes", {}); pa = p.get("attributes", {})
        for k in PRICE_LIKE:
            if k in ga and k in pa and is_price_obj(ga[k]) and is_price_obj(pa[k]):
                gcur, gval = norm_price(ga[k]); pcur, pval = norm_price(pa[k])
                if gcur == pcur:
                    diffs.append(abs(gval - pval))
    return {"price_mae": (sum(diffs)/len(diffs)) if diffs else 0.0}

def price_mape(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> Dict[str, float]:
    pred_map = {r["id"]: r for r in pred}
    parts = []
    for g in gold:
        p = pred_map.get(g["id"], {"attributes": {}})
        ga = g.get("attributes", {}); pa = p.get("attributes", {})
        for k in PRICE_LIKE:
            if k in ga and k in pa and is_price_obj(ga[k]) and is_price_obj(pa[k]):
                gcur, gval = norm_price(ga[k]); pcur, pval = norm_price(pa[k])
                if gcur == pcur and gval != 0:
                    parts.append(abs(gval - pval)/abs(gval))
    return {"price_mape": (sum(parts)/len(parts)) if parts else 0.0}
