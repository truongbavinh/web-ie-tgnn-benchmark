# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
from .utils import equal_attr_value

DEFAULT_REQUIRED = {
    "tourist": ["name","location","rating","price","duration"],
    "hotel": ["name","location","price","rating","amenities"],
    "realestate": ["title","location","price","area","bedrooms","bathrooms"],
    "flights": ["name","duration","stops","price","departure_time","arrival_time","airline"],
    "fashion": ["name","price"],
    "events": ["name","venue","date_time","artists"],
    "app": ["name","rating","category","developer","os"],
    "course": ["title","subject","fees","duration","instructor"],
    "scholarships": ["title","provider","amount","deadline","award"],
    "cooking": ["name","rating","author","time","type"],
}

def exact_match_record(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> Dict[str, float]:
    pred_map = {r["id"]: r for r in pred}
    ok = total = 0
    for g in gold:
        total += 1
        p = pred_map.get(g["id"], {"attributes": {}, "domain": g.get("domain")})
        domain = g.get("domain")
        req = DEFAULT_REQUIRED.get(domain, [])
        ga = g.get("attributes", {}); pa = p.get("attributes", {})
        all_ok = True
        for a in req:
            if a not in ga or a not in pa:
                all_ok = False; break
            if not equal_attr_value(a, ga[a], pa[a]):
                all_ok = False; break
        if all_ok: ok += 1
    return {"exact_match_record": ok/total if total else 0.0}
