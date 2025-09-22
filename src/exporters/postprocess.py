# -*- coding: utf-8 -*-
"""
Post-processing utilities to turn raw model outputs into final attributes matching schema.md.

Supported raw formats (per line JSON):
1) BIO spans (BERT token classification):
   {
     "id": "...",
     "domain": "fashion",
     "tokens": ["Striped","Shorts","..."],
     "tags":   ["B-NAME","I-NAME","..."]
   }

2) TGNN node predictions:
   {
     "id": "...",
     "domain": "fashion",
     "nodes": [
       {"idx": 0, "label": 3, "score": 0.91, "span": [s,e], "text": "cotton"}
       ...
     ],
     "label_map": { "0": "O", "1": "NAME", "2": "PRICE_VALUE", "3": "MATERIAL", ... }  # optional
   }

If 'label_map' is missing, you must provide it via function argument.
"""

from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import json, re

# ---------------- Domain registry (attributes & field types) -----------------

DOMAIN_SPEC = {
    "tourist":     {"required": ["name","location","rating","price","duration"], "optional": []},
    "hotel":       {"required": ["name","location","price","rating","amenities"], "optional": []},
    "realestate":  {"required": ["title","location","price","area","bedrooms","bathrooms"], "optional": []},
    "flights":     {"required": ["name","duration","stops","price","departure_time","arrival_time","airline"], "optional": []},
    "fashion":     {"required": ["name","price"], "optional": ["material","color","size"]},
    "events":      {"required": ["name","venue","date_time","artists"], "optional": []},
    "app":         {"required": ["name","rating","category","developer","os"], "optional": []},
    "course":      {"required": ["title","subject","fees","duration","instructor"], "optional": []},
    "scholarships":{"required": ["title","provider","amount","deadline","award"], "optional": []},
    "cooking":     {"required": ["name","rating","author","time","type"], "optional": []},
}

# ------------------------ Utilities ------------------------

def _merge_spans(tokens: List[str], tags: List[str]) -> Dict[str, List[str]]:
    """Collect spans by label from BIO tags at word level."""
    out: Dict[str, List[str]] = {}
    cur_label, cur_tokens = None, []
    def flush():
        nonlocal cur_label, cur_tokens
        if cur_label and cur_tokens:
            out.setdefault(cur_label, []).append(" ".join(cur_tokens).strip())
        cur_label, cur_tokens = None, []
    for tok, tag in zip(tokens, tags):
        if tag == "O" or not tag:
            flush(); continue
        if tag.startswith("B-"):
            flush(); cur_label = tag[2:]; cur_tokens = [tok]
        elif tag.startswith("I-") and cur_label == tag[2:]:
            cur_tokens.append(tok)
        else:
            # illegal transition -> flush and start new if I-*
            flush()
            if tag.startswith("I-"):
                cur_label = tag[2:]; cur_tokens = [tok]
    flush()
    return out

_money_num = re.compile(r"[0-9]+(?:[\.,][0-9]{1,2})?")
_currency = re.compile(r"(?i)(usd|eur|vnd|gbp|jpy|cny|krw|₫|\$|€|£|¥)")

def _parse_price_pieces(pieces: List[str]) -> Optional[Dict[str, Any]]:
    """Heuristic combination from spans to a price object {value,currency}."""
    if not pieces: return None
    joined = " ".join(pieces)
    cur = "USD"
    mcur = _currency.search(joined)
    if mcur:
        sym = mcur.group(1).lower()
        cur = {"$":"USD","usd":"USD","€":"EUR","eur":"EUR","£":"GBP","gbp":"GBP","¥":"JPY","jpy":"JPY","vnd":"VND","₫":"VND","krw":"KRW","cny":"CNY"}.get(sym, sym.upper())
    mnum = _money_num.search(joined.replace(",",""))
    if mnum:
        try:
            val = float(mnum.group(0))
            return {"value": val, "currency": cur}
        except Exception:
            return None
    return None

def _maybe_to_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None

# ------------------------ BIO -> attributes ------------------------

def attributes_from_bio(domain: str, tokens: List[str], tags: List[str]) -> Dict[str, Any]:
    spans_by_label = _merge_spans(tokens, tags)

    def first_or_none(label: str) -> Optional[str]:
        arr = spans_by_label.get(label, [])
        return arr[0] if arr else None

    attrs: Dict[str, Any] = {}

    if domain == "fashion":
        if (name := first_or_none("NAME")): attrs["name"] = name
        price_obj = None
        if (pv := first_or_none("PRICE_VALUE")) or (p := first_or_none("PRICE")):
            price_obj = _parse_price_pieces([pv or "", p or ""]) or _parse_price_pieces([pv or ""]) or _parse_price_pieces([p or ""]) 
        if price_obj: attrs["price"] = price_obj
        if (mats := spans_by_label.get("MATERIAL")): attrs["material"] = list(dict.fromkeys(mats))
        if (color := first_or_none("COLOR")): attrs["color"] = color
        if (sizes := spans_by_label.get("SIZE")): attrs["size"] = list(dict.fromkeys(sizes))

    elif domain == "hotel":
        if (name := first_or_none("NAME")): attrs["name"] = name
        if (loc := first_or_none("LOCATION")): attrs["location"] = loc
        if (rate := first_or_none("RATING")):
            try: attrs["rating"] = float(rate)
            except: pass
        if (price_obj := _parse_price_pieces(spans_by_label.get("PRICE", []))): attrs["price"] = price_obj
        if (ams := spans_by_label.get("AMENITY")): attrs["amenities"] = list(dict.fromkeys(ams))

    elif domain == "tourist":
        if (name := first_or_none("NAME")): attrs["name"] = name
        if (loc := first_or_none("LOCATION")): attrs["location"] = loc
        if (rate := first_or_none("RATING")):
            try: attrs["rating"] = float(rate)
            except: pass
        if (price_obj := _parse_price_pieces(spans_by_label.get("PRICE", []))): attrs["price"] = price_obj
        if (dur := first_or_none("DURATION")): attrs["duration"] = dur

    elif domain == "flights":
        if (name := first_or_none("ROUTE")) or (name := first_or_none("NAME")): attrs["name"] = name
        if (dur := first_or_none("DURATION")): attrs["duration"] = dur
        if (stops := first_or_none("STOPS")):
            iv = _maybe_to_int(stops);  attrs["stops"] = iv if iv is not None else 0
        if (price_obj := _parse_price_pieces(spans_by_label.get("PRICE", []))): attrs["price"] = price_obj
        if (dep := first_or_none("DEPARTURE_TIME")): attrs["departure_time"] = dep
        if (arr := first_or_none("ARRIVAL_TIME")): attrs["arrival_time"] = arr
        if (al := first_or_none("AIRLINE")): attrs["airline"] = al

    elif domain == "realestate":
        if (title := first_or_none("TITLE")): attrs["title"] = title
        if (loc := first_or_none("LOCATION")): attrs["location"] = loc
        if (price_obj := _parse_price_pieces(spans_by_label.get("PRICE", []))): attrs["price"] = price_obj
        if (area := first_or_none("AREA")):
            # naive parse "120 m2"
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(m2|sqft)", area, flags=re.I)
            if m:
                attrs["area"] = {"value": float(m.group(1)), "unit": m.group(2).lower()}
        if (bed := first_or_none("BEDROOMS")):
            iv = _maybe_to_int(bed);  attrs["bedrooms"] = iv if iv is not None else 0
        if (bath := first_or_none("BATHROOMS")):
            iv = _maybe_to_int(bath); attrs["bathrooms"] = iv if iv is not None else 0

    elif domain == "events":
        if (name := first_or_none("NAME")): attrs["name"] = name
        if (venue := first_or_none("VENUE")): attrs["venue"] = venue
        if (dt := first_or_none("DATE_TIME")): attrs["date_time"] = dt
        if (arts := spans_by_label.get("ARTIST")): attrs["artists"] = list(dict.fromkeys(arts))

    elif domain == "app":
        if (name := first_or_none("NAME")): attrs["name"] = name
        if (rate := first_or_none("RATING")):
            try: attrs["rating"] = float(rate)
            except: pass
        if (cat := first_or_none("CATEGORY")): attrs["category"] = cat
        if (dev := first_or_none("DEVELOPER")): attrs["developer"] = dev
        if (oss := spans_by_label.get("OS")): attrs["os"] = list(dict.fromkeys(oss))

    elif domain == "course":
        if (title := first_or_none("TITLE")): attrs["title"] = title
        if (subj := first_or_none("SUBJECT")): attrs["subject"] = subj
        if (fees := _parse_price_pieces(spans_by_label.get("FEES", []))): attrs["fees"] = fees
        if (dur := first_or_none("DURATION")): attrs["duration"] = dur
        if (ins := first_or_none("INSTRUCTOR")): attrs["instructor"] = ins

    elif domain == "scholarships":
        if (title := first_or_none("TITLE")): attrs["title"] = title
        if (prov := first_or_none("PROVIDER")): attrs["provider"] = prov
        if (amt := _parse_price_pieces(spans_by_label.get("AMOUNT", []))): attrs["amount"] = amt
        if (dl := first_or_none("DEADLINE")): attrs["deadline"] = dl
        if (award := first_or_none("AWARD")): attrs["award"] = award

    elif domain == "cooking":
        if (name := first_or_none("NAME")): attrs["name"] = name
        if (rate := first_or_none("RATING")):
            try: attrs["rating"] = float(rate)
            except: pass
        if (auth := first_or_none("AUTHOR")): attrs["author"] = auth
        if (t := first_or_none("TIME")): attrs["time"] = t
        if (typ := first_or_none("TYPE")): attrs["type"] = typ

    # Drop keys with None
    return {k:v for k,v in attrs.items() if v is not None}

# ------------------------ TGNN nodes -> attributes ------------------------

def attributes_from_graph(domain: str, nodes: List[Dict[str, Any]], label_map: Optional[Dict[str,str]] = None) -> Dict[str, Any]:
    """Nodes contain label ids; map to slot names using label_map then aggregate."""
    # map label ids -> names
    def to_name(lid: int) -> str:
        if label_map is None: return str(lid)
        return label_map.get(str(lid), label_map.get(int(lid), str(lid))) if isinstance(label_map, dict) else str(lid)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        name = to_name(n.get("label", -1))
        if name == "O":
            continue
        buckets.setdefault(name, []).append(n)

    # Convert buckets to attributes using heuristics similar to BIO
    attrs: Dict[str, Any] = {}

    def first_text(label: str) -> Optional[str]:
        arr = buckets.get(label, [])
        for x in arr:
            t = x.get("text")
            if t: return t
        return None

    if domain == "fashion":
        if (name := first_text("NAME")): attrs["name"] = name
        if "PRICE" in buckets or "PRICE_VALUE" in buckets:
            pieces = [x.get("text", "") for x in buckets.get("PRICE", [])] + [x.get("text", "") for x in buckets.get("PRICE_VALUE", [])]
            price_obj = _parse_price_pieces([p for p in pieces if p])
            if price_obj: attrs["price"] = price_obj
        mats = [x.get("text") for x in buckets.get("MATERIAL", []) if x.get("text")]
        if mats: attrs["material"] = list(dict.fromkeys(mats))
        if (c := first_text("COLOR")): attrs["color"] = c
        sizes = [x.get("text") for x in buckets.get("SIZE", []) if x.get("text")]
        if sizes: attrs["size"] = list(dict.fromkeys(sizes))

    # (For brevity, other domains can mirror attributes_from_bio heuristics)
    # You can extend similarly if your graph predicts fine-grained labels per domain.

    return {k:v for k,v in attrs.items() if v is not None}

# ------------------------ Dispatcher ------------------------

def to_attributes(raw_path: str, domain: str, out_path: str, label_map: Optional[Dict[str,str]] = None) -> None:
    """Read raw JSONL and write predictions.jsonl (id, domain, attributes)."""
    with open(raw_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            rid = r.get("id"); rdom = r.get("domain", domain)
            attrs: Dict[str, Any] = {}
            if "tokens" in r and "tags" in r:
                attrs = attributes_from_bio(rdom, r["tokens"], r["tags"])  # BERT/BIO
            elif "nodes" in r:
                lm = r.get("label_map", label_map)
                attrs = attributes_from_graph(rdom, r["nodes"], lm)  # TGNN
            else:
                # Unknown format -> skip
                attrs = {}

            pred = {"id": rid, "domain": rdom, "attributes": attrs}
            out.write(json.dumps(pred, ensure_ascii=False) + "\n")
