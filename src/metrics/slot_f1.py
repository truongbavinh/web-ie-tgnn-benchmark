# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple, List
from .utils import instances_from_attributes

def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r / (p + r) if (p + r) else 0.0
    return p, r, f1

def f1_slot_micro(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> Dict[str, float]:
    pred_map = {r["id"]: r for r in pred}
    tp = fp = fn = 0
    for g in gold:
        p = pred_map.get(g["id"], {"attributes": {}})
        gset = instances_from_attributes(g.get("attributes", {}))
        pset = instances_from_attributes(p.get("attributes", {}))
        tp += len(gset & pset); fp += len(pset - gset); fn += len(gset - pset)
    P,R,F = _prf(tp,fp,fn)
    return {"f1_slot_micro": F, "precision": P, "recall": R}

def f1_slot_macro(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> Dict[str, float]:
    pred_map = {r["id"]: r for r in pred}
    from collections import defaultdict
    agg = defaultdict(lambda: [0,0,0])  # attr -> [tp,fp,fn]
    for g in gold:
        p = pred_map.get(g["id"], {"attributes": {}})
        gset = instances_from_attributes(g.get("attributes", {}))
        pset = instances_from_attributes(p.get("attributes", {}))
        g_by = defaultdict(set); p_by = defaultdict(set)
        for a,v in gset: g_by[a].add(v)
        for a,v in pset: p_by[a].add(v)
        for a in set(g_by.keys()) | set(p_by.keys()):
            gi = g_by.get(a,set()); pi = p_by.get(a,set())
            tp = len(gi & pi); fp = len(pi - gi); fn = len(gi - pi)
            agg[a][0]+=tp; agg[a][1]+=fp; agg[a][2]+=fn
    f1s = []
    for a,(tp,fp,fn) in agg.items():
        _,_,f = _prf(tp,fp,fn); f1s.append(f)
    macro = sum(f1s)/len(f1s) if f1s else 0.0
    return {"f1_slot_macro": macro}
