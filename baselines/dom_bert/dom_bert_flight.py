#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DOM+BERT for Flight IE (NAME, DURATION, STOPS, PRICE, DEPARTURE_TIME, ARRIVAL_TIME, AIRLINE)
- DOM-first regex backoff:
    * NAME: <title>, og:title, h1, JSON-LD name/headline
    * AIRLINE: JSON-LD airline/carrier, meta, table/dl rows, "Airline: ..."
    * DEPARTURE_TIME: JSON-LD departureTime/startDate, <time datetime>, text patterns (ISO, "Jan 4, 2026 7:00 PM")
    * ARRIVAL_TIME: JSON-LD arrivalTime/endDate, <time datetime>, text patterns
    * DURATION: JSON-LD duration/flightDuration, ISO8601 (PT2H35M), "2h 35m", "2 hours 35 minutes"
    * STOPS: JSON-LD stops/numberOfStops, "Nonstop", "Direct", "1 stop", "2 stops"
    * PRICE: JSON-LD offers/price, meta product:price:amount, regex currency "$199" / "USD 199" / "199,000 VND"
- Presence metric giống domain App/Course (ALL7)
"""

import os, re, csv, json, glob, math, hashlib, html, pathlib, random, statistics
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_linear_schedule_with_warmup

from bs4 import BeautifulSoup
from bs4.element import Tag
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DOMAIN         = "Flight"
HTML_DIR       = "flights_html"
MERGED_GOLD    = "flight_gt_merged.json"
OUT_DIR        = "dombert_out"
BASE_MODEL     = "bert-base-uncased"

SEEDS          = [42, 43, 44]
TRAIN_RATIO    = 0.7
VAL_RATIO      = 0.1
MAX_EPOCHS     = 12
BATCH_SIZE     = 16
LR             = 5e-5
WARMUP_RATIO   = 0.1
MAX_TOKENS     = 512

SAVE_CKPT_DIR  = "dombert_ckpts"
USE_REGEX_BACKOFF = True

# Early Stopping
ES_MONITOR     = "presence"
ES_PATIENCE    = 3
ES_MIN_DELTA   = 0.002

# Label space (BIO)
LABELS = [
    "O",
    "B-NAME","I-NAME",
    "B-DURATION","I-DURATION",
    "B-STOPS","I-STOPS",
    "B-PRICE","I-PRICE",
    "B-DEPARTURE_TIME","I-DEPARTURE_TIME",
    "B-ARRIVAL_TIME","I-ARRIVAL_TIME",
    "B-AIRLINE","I-AIRLINE",
]
LABEL2ID = {lab:i for i,lab in enumerate(LABELS)}
ID2LABEL = {i:lab for lab,i in LABEL2ID.items()}

PRESENCE_LABELS = ["NAME","DURATION","STOPS","PRICE","DEPARTURE_TIME","ARRIVAL_TIME","AIRLINE"]

# =========================
# Utils
# =========================
def list_html_files(root: str) -> List[str]:
    pats = [os.path.join(root, "**", "*.html"), os.path.join(root, "**", "*.htm")]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files

def deterministic_split(paths: List[str], seed: int, train_ratio: float=0.7):
    scored = []
    for fp in paths:
        h = hashlib.md5((str(seed) + "|" + fp).encode("utf-8")).hexdigest()
        k = int(h[:8], 16)
        scored.append((k, fp))
    scored.sort(key=lambda x: x[0])
    n_train = int(math.floor(len(scored) * train_ratio))
    train = [fp for _, fp in scored[:n_train]]
    test  = [fp for _, fp in scored[n_train:]]
    return train, test

def deterministic_subsplit(paths: List[str], seed: int, val_ratio: float):
    scored = []
    for fp in paths:
        h = hashlib.md5((f"VAL|{seed}|{fp}").encode("utf-8")).hexdigest()
        k = int(h[:8], 16)
        scored.append((k, fp))
    scored.sort(key=lambda x: x[0])
    n_val = int(math.floor(len(scored) * val_ratio))
    val = [fp for _, fp in scored[:n_val]]
    train_sub = [fp for _, fp in scored[n_val:]]
    return train_sub, val

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def strip_html_to_text(raw_html: str) -> str:
    txt = re.sub(r"<!--.*?-->", " ", raw_html, flags=re.DOTALL)
    txt = re.sub(r"<script.*?>.*?</script>", " ", txt, flags=re.DOTALL|re.IGNORECASE)
    txt = re.sub(r"<style.*?>.*?</style>",  " ", txt, flags=re.DOTALL|re.IGNORECASE)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def normalize_text(s: str) -> str:
    return normalize_space(s).lower()

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# =========================
# Gold loading
# =========================
MERGED_LABEL_MAP = {
    "name":"NAME",
    "flight_name":"NAME",
    "title":"NAME",
    "route":"NAME",

    "duration":"DURATION",
    "flight_duration":"DURATION",
    "time":"DURATION",

    "stops":"STOPS",
    "stop":"STOPS",
    "number_of_stops":"STOPS",
    "numberofstops":"STOPS",
    "layovers":"STOPS",
    "layover":"STOPS",

    "price":"PRICE",
    "fare":"PRICE",
    "cost":"PRICE",
    "amount":"PRICE",

    "departure_time":"DEPARTURE_TIME",
    "departure":"DEPARTURE_TIME",
    "depart_time":"DEPARTURE_TIME",
    "depart":"DEPARTURE_TIME",
    "start_time":"DEPARTURE_TIME",
    "startdate":"DEPARTURE_TIME",
    "start_date":"DEPARTURE_TIME",

    "arrival_time":"ARRIVAL_TIME",
    "arrival":"ARRIVAL_TIME",
    "arrive_time":"ARRIVAL_TIME",
    "arrive":"ARRIVAL_TIME",
    "end_time":"ARRIVAL_TIME",
    "enddate":"ARRIVAL_TIME",
    "end_date":"ARRIVAL_TIME",

    "airline":"AIRLINE",
    "carrier":"AIRLINE",
    "operated_by":"AIRLINE",
    "operator":"AIRLINE",
}

def load_merged_gold(path: str) -> Dict[str, Dict[str, List[str]]]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[str, Dict[str, List[str]]] = {}
    for html_name, items in (obj or {}).items():
        base = os.path.basename(html_name)
        bucket = defaultdict(list)
        for _id, rec in (items or {}).items():
            lab_raw = str(rec.get("label","")).strip().lower()
            txt_raw = (rec.get("text") or "").strip()
            if not lab_raw or not txt_raw:
                continue
            lab = MERGED_LABEL_MAP.get(lab_raw)
            if not lab:
                continue
            bucket[lab].append(normalize_space(txt_raw))
        if bucket:
            out[base] = dict(bucket)
    return out

# =========================
# Flight normalizers + regex/DOM backoff
# =========================
AIRLINE_LABEL = re.compile(r"^\s*(airline|carrier|operated\s*by|operator)\s*:?\s*$", re.I)
DEP_LABEL     = re.compile(r"^\s*(departure|depart|departs?|departure\s*time|depart\s*time|take\s*off|takeoff|start)\s*:?\s*$", re.I)
ARR_LABEL     = re.compile(r"^\s*(arrival|arrive|arrives?|arrival\s*time|arrive\s*time|landing|end)\s*:?\s*$", re.I)
DUR_LABEL     = re.compile(r"^\s*(duration|flight\s*duration|travel\s*time|elapsed)\s*:?\s*$", re.I)
STOPS_LABEL   = re.compile(r"^\s*(stops?|stopovers?|layovers?)\s*:?\s*$", re.I)
PRICE_LABEL   = re.compile(r"^\s*(price|fare|cost|total|from)\s*:?\s*$", re.I)

# time patterns (reuse event DT but add "HH:MM" only cases)
ISO_DT = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?(?:Z|[+-]\d{2}:\d{2})?\b")
TEXT_DT = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{2,4})"
    r"(?:\s*(?:at|@)?\s*\d{1,2}:\d{2}(?:\s*(?:am|pm|AM|PM))?)?\b"
)
TIME_ONLY = re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b")

# duration patterns
ISO8601_DUR = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$", re.I)
DUR_HM = re.compile(r"\b(\d{1,2})\s*(?:h|hr|hrs|hour|hours)\b(?:\s*(\d{1,2})\s*(?:m|min|mins|minute|minutes)\b)?", re.I)
DUR_MIN = re.compile(r"\b(\d{1,4})\s*(?:m|min|mins|minute|minutes)\b", re.I)

# price patterns (same as course/fashion)
CURRENCY = r"(?:\$|usd|us\$|vnd|đ|₫|eur|€|gbp|£|aud|cad|sgd|inr|₹|jpy|¥|krw|₩)"
MONEY_RE = re.compile(
    rf"(?<!\w)({CURRENCY})\s*([0-9]{{1,3}}(?:[,\.\s][0-9]{{3}})*(?:\.[0-9]{{1,2}})?)\b",
    re.I
)
MONEY_AFTER = re.compile(
    rf"\b([0-9]{{1,3}}(?:[,\.\s][0-9]{{3}})*(?:\.[0-9]{{1,2}})?)\s*({CURRENCY})(?!\w)",
    re.I
)

# stops patterns
NONSTOP_RE = re.compile(r"\b(non[\s-]?stop|nonstop|direct)\b", re.I)
STOP_RE = re.compile(r"\b(\d{1,2})\s*(stop|stops|stopover|stopovers|layover|layovers)\b", re.I)

def _parse_iso8601_duration(d: str) -> Optional[str]:
    d = (d or "").strip()
    if not d:
        return None
    m = ISO8601_DUR.match(d)
    if not m:
        return None
    days = int(m.group(1) or 0)
    hrs  = int(m.group(2) or 0)
    mins = int(m.group(3) or 0)
    secs = int(m.group(4) or 0)
    parts = []
    if days: parts.append(f"{days} d")
    if hrs:  parts.append(f"{hrs} h")
    if mins: parts.append(f"{mins} min")
    if secs and not parts:
        parts.append(f"{secs} s")
    return " ".join(parts) if parts else None

def normalize_duration(s: str) -> Optional[str]:
    t = (s or "").strip()
    if not t:
        return None
    iso = _parse_iso8601_duration(t)
    if iso:
        return iso
    m = DUR_HM.search(t)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2) or 0)
        if mm:
            return f"{h} h {mm} min"
        return f"{h} h"
    m2 = DUR_MIN.search(t)
    if m2:
        mm = int(m2.group(1))
        return f"{mm} min"
    # fallback: if contains "duration" nearby, try extract digits
    if re.search(r"\b(duration|flight\s*duration|travel\s*time)\b", t, re.I):
        nm = re.search(r"\b\d{1,2}\b", t)
        if nm:
            return nm.group(0)
    return None

def normalize_price(s: str) -> Optional[str]:
    t = (s or "").strip()
    if not t:
        return None
    m = MONEY_RE.search(t)
    if m:
        cur = m.group(1).upper()
        amt = re.sub(r"\s+", "", m.group(2))
        return f"{cur} {amt}"
    m2 = MONEY_AFTER.search(t)
    if m2:
        amt = re.sub(r"\s+", "", m2.group(1))
        cur = m2.group(2).upper()
        return f"{cur} {amt}"
    if re.search(r"\b(price|fare|cost|total)\b", t, re.I):
        nm = re.search(r"\b\d{1,3}(?:[,\.\s]\d{3})*(?:\.\d{1,2})?\b", t)
        if nm:
            return nm.group(0)
    return None

def normalize_stops(s: str) -> Optional[str]:
    t = normalize_space(s)
    if not t:
        return None
    if NONSTOP_RE.search(t):
        return "0"
    m = STOP_RE.search(t)
    if m:
        return str(int(m.group(1)))
    # sometimes "1-stop" or "2x stops"
    m2 = re.search(r"\b(\d{1,2})\s*[- ]?\s*stop\b", t, re.I)
    if m2:
        return str(int(m2.group(1)))
    return None

def normalize_time_value(s: str) -> Optional[str]:
    t = (s or "").strip()
    if not t:
        return None
    m = ISO_DT.search(t)
    if m:
        return m.group(0)
    m2 = TEXT_DT.search(t)
    if m2:
        return m2.group(0)
    m3 = TIME_ONLY.search(t)
    if m3:
        return m3.group(0)
    if re.match(r"^\d{4}-\d{2}-\d{2}", t):
        return t
    return None

def normalize_airline(s: str) -> Optional[str]:
    t = normalize_space(s)
    if not t:
        return None
    # avoid huge blocks
    t = t[:120]
    # strip "Operated by"
    t = re.sub(r"^\s*(operated\s*by|carrier)\s*[:\-–]\s*", "", t, flags=re.I).strip()
    return t if 2 <= len(t) <= 120 else None

def _rows_from_soup(soup: BeautifulSoup) -> List[Tuple[str,str]]:
    rows=[]
    for tr in soup.find_all("tr"):
        th = tr.find(["th","td"])
        tds = tr.find_all("td")
        if th and tds:
            lab = _clean(th.get_text(" ", strip=True))
            val = _clean(" ".join(td.get_text(" ", strip=True) for td in tds[1:]) or (tds[-1].get_text(" ", strip=True) if tds else ""))
            rows.append((lab, val))
    for dl in soup.find_all("dl"):
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                rows.append((_clean(dt.get_text(" ", strip=True)), _clean(dd.get_text(" ", strip=True))))
    return rows

def _jsonld_find_first(soup: BeautifulSoup, keys: List[str]) -> Optional[str]:
    for sc in soup.find_all("script", attrs={"type":"application/ld+json"}):
        raw = (sc.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        def visit(obj):
            if isinstance(obj, dict):
                for k in keys:
                    if k in obj:
                        v = obj[k]
                        if isinstance(v, (str,int,float)):
                            return str(v)
                        if isinstance(v, dict):
                            for kk in ["name","text","value","price","currency","departureTime","arrivalTime","duration"]:
                                if kk in v and isinstance(v[kk], (str,int,float)):
                                    return str(v[kk])
                            return json.dumps(v, ensure_ascii=False)
                        if isinstance(v, list):
                            return " ".join(str(x) for x in v[:8])
                for vv in obj.values():
                    r = visit(vv)
                    if r: return r
            elif isinstance(obj, list):
                for it in obj:
                    r = visit(it)
                    if r: return r
            return None

        r = visit(data)
        if r:
            return _clean(r)
    return None

# =========================
# DOM finders
# =========================
def bs4_find_name_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    og = soup.find("meta", attrs={"property":"og:title"})
    if og and og.get("content"):
        cands.append(_clean(og["content"]))

    if soup.title and soup.title.get_text(strip=True):
        cands.append(_clean(soup.title.get_text(" ", strip=True)))

    h1 = soup.find("h1")
    if h1:
        cands.append(_clean(h1.get_text(" ", strip=True)))

    jn = _jsonld_find_first(soup, ["name","headline"])
    if jn:
        cands.append(_clean(jn))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_airline_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    ja = _jsonld_find_first(soup, ["airline","carrier"])
    if ja:
        na = normalize_airline(ja) or ja
        cands.append(_clean(na))

    # some pages use "provider"/"operator"
    jo = _jsonld_find_first(soup, ["provider","operator"])
    if jo:
        na = normalize_airline(jo) or jo
        cands.append(_clean(na))

    rows = _rows_from_soup(soup)
    for lab, val in rows:
        if AIRLINE_LABEL.match(lab) and val:
            na = normalize_airline(val) or val
            cands.append(_clean(na))

    full = _clean(soup.get_text(" ", strip=True))
    m = re.search(r"\b(Airline|Carrier|Operated\s*by|Operator)\s*[:\-–]\s*([^|•;\n]{2,120})", full, flags=re.I)
    if m:
        na = normalize_airline(m.group(2)) or m.group(2)
        cands.append(_clean(na))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v or len(v) > 140:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_departure_time_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    jd = _jsonld_find_first(soup, ["departureTime","startDate","departure"])
    if jd:
        nd = normalize_time_value(jd) or jd
        cands.append(_clean(nd))

    for ttag in soup.find_all("time"):
        dt = (ttag.get("datetime") or "").strip()
        if dt:
            nd = normalize_time_value(dt)
            if nd:
                cands.append(_clean(nd))
                break

    rows = _rows_from_soup(soup)
    for lab, val in rows:
        if DEP_LABEL.match(lab) and val:
            nd = normalize_time_value(val) or val
            cands.append(_clean(nd))

    full = _clean(soup.get_text(" ", strip=True))
    m = re.search(r"\b(Departure|Departs?|Departure\s*Time|Depart\s*Time|Take\s*off|Start)\s*[:\-–]\s*([^|•;\n]{2,160})",
                  full, flags=re.I)
    if m:
        nd = normalize_time_value(m.group(2)) or m.group(2)
        cands.append(_clean(nd))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_arrival_time_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    ja = _jsonld_find_first(soup, ["arrivalTime","endDate","arrival"])
    if ja:
        na = normalize_time_value(ja) or ja
        cands.append(_clean(na))

    for ttag in soup.find_all("time"):
        dt = (ttag.get("datetime") or "").strip()
        if dt:
            na = normalize_time_value(dt)
            if na:
                cands.append(_clean(na))
                break

    rows = _rows_from_soup(soup)
    for lab, val in rows:
        if ARR_LABEL.match(lab) and val:
            na = normalize_time_value(val) or val
            cands.append(_clean(na))

    full = _clean(soup.get_text(" ", strip=True))
    m = re.search(r"\b(Arrival|Arrives?|Arrival\s*Time|Arrive\s*Time|Landing|End)\s*[:\-–]\s*([^|•;\n]{2,160})",
                  full, flags=re.I)
    if m:
        na = normalize_time_value(m.group(2)) or m.group(2)
        cands.append(_clean(na))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_duration_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    jd = _jsonld_find_first(soup, ["duration","flightDuration","travelTime"])
    if jd:
        nd = normalize_duration(jd) or jd
        cands.append(_clean(nd))

    rows = _rows_from_soup(soup)
    for lab, val in rows:
        if DUR_LABEL.match(lab) and val:
            nd = normalize_duration(val) or val
            cands.append(_clean(nd))

    full = _clean(soup.get_text(" ", strip=True))
    m = re.search(r"\b(Duration|Flight\s*Duration|Travel\s*Time|Elapsed)\s*[:\-–]\s*([^|•;\n]{2,120})", full, flags=re.I)
    if m:
        nd = normalize_duration(m.group(2)) or m.group(2)
        cands.append(_clean(nd))
    else:
        nd = normalize_duration(full)
        if nd:
            cands.append(_clean(nd))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_stops_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    js = _jsonld_find_first(soup, ["stops","numberOfStops","stopovers","layovers"])
    if js:
        ns = normalize_stops(js) or js
        cands.append(_clean(ns))

    rows = _rows_from_soup(soup)
    for lab, val in rows:
        if STOPS_LABEL.match(lab) and val:
            ns = normalize_stops(val) or val
            cands.append(_clean(ns))

    full = _clean(soup.get_text(" ", strip=True))
    if NONSTOP_RE.search(full):
        cands.append("0")
    else:
        m = STOP_RE.search(full)
        if m:
            cands.append(str(int(m.group(1))))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def bs4_find_price_candidates(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cands = []

    jo = _jsonld_find_first(soup, ["price","offers","fare"])
    if jo:
        np = normalize_price(jo) or jo
        cands.append(_clean(np))

    mp = soup.find("meta", attrs={"property":re.compile(r"(product:price:amount|og:price:amount)", re.I)})
    if mp and mp.get("content"):
        np = normalize_price(mp["content"]) or mp["content"]
        cands.append(_clean(np))

    rows = _rows_from_soup(soup)
    for lab, val in rows:
        if PRICE_LABEL.match(lab) and val:
            np = normalize_price(val) or val
            cands.append(_clean(np))

    full = _clean(soup.get_text(" ", strip=True))
    m = re.search(r"\b(Price|Fare|Cost|Total|From)\s*[:\-–]\s*([^|•;\n]{2,120})", full, flags=re.I)
    if m:
        np = normalize_price(m.group(2)) or m.group(2)
        cands.append(_clean(np))
    else:
        np = normalize_price(full)
        if np:
            cands.append(_clean(np))

    out, seen = [], set()
    for v in cands:
        v = v.strip(" -|•\t\r\n")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
        if len(out) >= 2:
            break
    return out

def regex_backoff_candidates(text: str, raw_html: Optional[str]=None) -> Dict[str, List[str]]:
    """
    DOM-first (bs4_*), fallback regex from plain text.
    """
    res: Dict[str, List[str]] = {}

    if raw_html:
        n = bs4_find_name_candidates(raw_html)
        if n: res["NAME"] = n[:2]
        d = bs4_find_duration_candidates(raw_html)
        if d: res["DURATION"] = d[:2]
        s = bs4_find_stops_candidates(raw_html)
        if s: res["STOPS"] = s[:2]
        p = bs4_find_price_candidates(raw_html)
        if p: res["PRICE"] = p[:2]
        dep = bs4_find_departure_time_candidates(raw_html)
        if dep: res["DEPARTURE_TIME"] = dep[:2]
        arr = bs4_find_arrival_time_candidates(raw_html)
        if arr: res["ARRIVAL_TIME"] = arr[:2]
        al = bs4_find_airline_candidates(raw_html)
        if al: res["AIRLINE"] = al[:2]

    plain = text or ""

    if "NAME" not in res:
        head = plain[:200]
        m = re.search(r"([A-Za-z0-9][A-Za-z0-9 \-\–\—\:\|•]{3,120})", head)
        if m:
            res["NAME"] = [_clean(m.group(1).split("|")[0])]

    if "AIRLINE" not in res:
        m = re.search(r"\b(Airline|Carrier|Operated\s*by|Operator)\s*[:\-–]\s*([^|•;\n]{2,120})", plain, flags=re.I)
        if m:
            na = normalize_airline(m.group(2)) or m.group(2)
            res["AIRLINE"] = [_clean(na)]

    if "DEPARTURE_TIME" not in res:
        m = re.search(r"\b(Departure|Departs?|Departure\s*Time|Depart\s*Time|Take\s*off|Start)\s*[:\-–]\s*([^|•;\n]{2,160})", plain, flags=re.I)
        if m:
            nd = normalize_time_value(m.group(2)) or m.group(2)
            res["DEPARTURE_TIME"] = [_clean(nd)]
        else:
            nd = normalize_time_value(plain)
            if nd:
                res["DEPARTURE_TIME"] = [nd]

    if "ARRIVAL_TIME" not in res:
        m = re.search(r"\b(Arrival|Arrives?|Arrival\s*Time|Arrive\s*Time|Landing|End)\s*[:\-–]\s*([^|•;\n]{2,160})", plain, flags=re.I)
        if m:
            na = normalize_time_value(m.group(2)) or m.group(2)
            res["ARRIVAL_TIME"] = [_clean(na)]

    if "DURATION" not in res:
        m = re.search(r"\b(Duration|Flight\s*Duration|Travel\s*Time|Elapsed)\s*[:\-–]\s*([^|•;\n]{2,120})", plain, flags=re.I)
        if m:
            nd = normalize_duration(m.group(2)) or m.group(2)
            res["DURATION"] = [_clean(nd)]
        else:
            nd = normalize_duration(plain)
            if nd:
                res["DURATION"] = [nd]

    if "STOPS" not in res:
        if NONSTOP_RE.search(plain):
            res["STOPS"] = ["0"]
        else:
            m = STOP_RE.search(plain)
            if m:
                res["STOPS"] = [str(int(m.group(1)))]

    if "PRICE" not in res:
        m = re.search(r"\b(Price|Fare|Cost|Total|From)\s*[:\-–]\s*([^|•;\n]{2,120})", plain, flags=re.I)
        if m:
            np = normalize_price(m.group(2)) or m.group(2)
            res["PRICE"] = [_clean(np)]
        else:
            np = normalize_price(plain)
            if np:
                res["PRICE"] = [np]

    return res

# =========================
# Dataset & Model (BIO)
# =========================
@dataclass
class Example:
    text: str
    tokens: List[str]
    input_ids: List[int]
    attention_mask: List[int]
    offset_mapping: List[Tuple[int,int]]
    labels: Optional[List[int]] = None
    file: str = ""

class DomBertDataset(Dataset):
    def __init__(self, files: List[str], tokenizer, split: str, gold_merged: Dict[str, Dict[str, List[str]]]):
        self.items: List[Example] = []
        for fp in tqdm(files, desc=f"Build {split}"):
            raw = read_file(fp)
            flat = strip_html_to_text(raw)
            if not flat:
                continue
            enc = tokenizer(flat, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=MAX_TOKENS)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            offs = enc["offset_mapping"]
            labels = [LABEL2ID["O"]] * len(input_ids)

            base = os.path.basename(fp)
            gold = gold_merged.get(base, {})

            def mark_spans(lst: List[str], ent: str):
                for gtxt in lst:
                    gnorm = normalize_text(gtxt)
                    if not gnorm:
                        continue
                    for m in re.finditer(re.escape(gnorm), normalize_text(flat)):
                        gs, ge = m.span()
                        started = False
                        for i, (s, e) in enumerate(offs):
                            if s == 0 and e == 0:
                                continue
                            if e <= gs or s >= ge:
                                continue
                            lab = f"I-{ent}"
                            if not started:
                                lab = f"B-{ent}"
                                started = True
                            labels[i] = LABEL2ID.get(lab, LABEL2ID["O"])

            for lab in PRESENCE_LABELS:
                vals = gold.get(lab, [])
                mark_spans(vals, lab)

            self.items.append(
                Example(
                    flat,
                    tokenizer.convert_ids_to_tokens(input_ids),
                    input_ids,
                    attn,
                    [(int(s), int(e)) for (s, e) in offs],
                    labels,
                    fp
                )
            )

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

class DomBert(nn.Module):
    def __init__(self, base_model: str, num_labels: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        self.bert   = AutoModel.from_pretrained(base_model, config=self.config)
        H = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.cls     = nn.Linear(H, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.cls(self.dropout(out))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

def collate(batch: List[Example]):
    input_ids = torch.tensor([ex.input_ids for ex in batch], dtype=torch.long)
    attn      = torch.tensor([ex.attention_mask for ex in batch], dtype=torch.long)
    labels    = torch.tensor([ex.labels for ex in batch], dtype=torch.long) if batch[0].labels is not None else None
    meta      = [{"file": ex.file, "offsets": ex.offset_mapping, "text": ex.text} for ex in batch]
    return input_ids, attn, labels, meta

# =========================
# Decode & Metrics
# =========================
def decode_bio_to_sets(meta, logits) -> Dict[str, List[str]]:
    text = meta["text"]
    offsets = meta["offsets"]
    pred_ids = logits.argmax(-1).tolist()

    spans = []
    cur_lab=None; cur_s=None; cur_e=None

    for pid, (s,e) in zip(pred_ids, offsets):
        if s == 0 and e == 0:
            if cur_lab is not None:
                spans.append((cur_lab, cur_s, cur_e))
                cur_lab=None
            continue

        lab = ID2LABEL.get(pid, "O")
        if lab == "O":
            if cur_lab is not None:
                spans.append((cur_lab, cur_s, cur_e))
                cur_lab=None
            continue

        if lab.startswith("B-"):
            if cur_lab is not None:
                spans.append((cur_lab, cur_s, cur_e))
            cur_lab = lab[2:]
            cur_s = s
            cur_e = e
        elif lab.startswith("I-"):
            l2 = lab[2:]
            if cur_lab == l2:
                cur_e = e
            else:
                cur_lab = l2
                cur_s = s
                cur_e = e

    if cur_lab is not None:
        spans.append((cur_lab, cur_s, cur_e))

    out = defaultdict(list)
    for lab, s, e in spans:
        val = normalize_space(text[s:e])

        if lab == "PRICE":
            nv = normalize_price(val)
            if nv: val = nv

        if lab == "DURATION":
            nv = normalize_duration(val)
            if nv: val = nv

        if lab == "STOPS":
            nv = normalize_stops(val)
            if nv: val = nv

        if lab in ("DEPARTURE_TIME","ARRIVAL_TIME"):
            nv = normalize_time_value(val)
            if nv: val = nv

        if lab == "AIRLINE":
            nv = normalize_airline(val)
            if nv: val = nv

        out[lab].append(val)

    return {k: sorted(set(v)) for k, v in out.items()}

def eval_presence_for_items(items: List[Example], perfile_preds: Dict[str, Dict[str, List[str]]], gold_merged: Dict[str, Dict[str, List[str]]]):
    n = 0
    sums = {k:0 for k in PRESENCE_LABELS}
    all7_sum = 0

    for ex in items:
        base = os.path.basename(ex.file)
        gold = gold_merged.get(base)
        if gold is None:
            continue
        preds = perfile_preds.get(ex.file, {})

        g = {lab: bool(gold.get(lab)) for lab in PRESENCE_LABELS}
        p = {lab: bool(preds.get(lab)) for lab in PRESENCE_LABELS}

        ok_all = True
        for lab in PRESENCE_LABELS:
            score = 100 if (g[lab] == p[lab]) else 0
            sums[lab] += score
            if score != 100:
                ok_all = False

        all7_sum += 100 if ok_all else 0
        n += 1

    if n == 0:
        return {lab:0.0 for lab in PRESENCE_LABELS} | {"ALL7":0.0}

    out = {lab: (sums[lab] / n) for lab in PRESENCE_LABELS}
    out["ALL7"] = all7_sum / n
    return out

# =========================
# Train / Infer
# =========================
def run_inference_collect_preds(model, loader, device) -> Dict[str, Dict[str, List[str]]]:
    perfile = {}
    model.eval()
    with torch.no_grad():
        for input_ids, attn, labels, meta in loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            logits, _ = model(input_ids=input_ids, attention_mask=attn, labels=None)
            dec = decode_bio_to_sets(meta[0], logits[0].cpu())

            if USE_REGEX_BACKOFF:
                raw = read_file(meta[0]["file"]) if os.path.isfile(meta[0]["file"]) else ""
                plain = strip_html_to_text(raw)
                rb = regex_backoff_candidates(plain, raw_html=raw)
                for lab in PRESENCE_LABELS:
                    if not dec.get(lab) and rb.get(lab):
                        dec[lab] = rb[lab]

            perfile[meta[0]["file"]] = dec
    return perfile

def train_one_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    files = list_html_files(HTML_DIR)
    if not files:
        raise RuntimeError(f"No HTML in {HTML_DIR}")

    train_files, test_files = deterministic_split(files, seed, TRAIN_RATIO)
    train_sub, val_files = deterministic_subsplit(train_files, seed, VAL_RATIO)

    gold_merged = load_merged_gold(MERGED_GOLD)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    ds_train = DomBertDataset(train_sub, tokenizer, "train", gold_merged)
    ds_val   = DomBertDataset(val_files,  tokenizer, "val",   gold_merged)
    ds_test  = DomBertDataset(test_files, tokenizer, "test",  gold_merged)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomBert(BASE_MODEL, num_labels=len(LABELS)).to(device)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(ds_val,   batch_size=1, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(ds_test,  batch_size=1, shuffle=False, collate_fn=collate)

    if MAX_EPOCHS > 0:
        t_steps = max(1, len(train_loader) * MAX_EPOCHS)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        sch = get_linear_schedule_with_warmup(opt, int(WARMUP_RATIO * t_steps), t_steps)

        best_metric = None
        best_state = None
        patience = 0

        for ep in range(1, MAX_EPOCHS + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Seed {seed} Train ep{ep}")
            for input_ids, attn, labels, meta in pbar:
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)

                opt.zero_grad()
                _, loss = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                pbar.set_postfix(loss=float(loss))

            val_preds = run_inference_collect_preds(model, val_loader, device)
            pres = eval_presence_for_items(ds_val.items, val_preds, gold_merged)
            current = pres["ALL7"] if ES_MONITOR == "presence" else 0.0
            print(f"[VAL] ep{ep} presence " +
                  "/".join([f"{k}={pres[k]:.2f}" for k in PRESENCE_LABELS]) +
                  f" / ALL7={pres['ALL7']:.2f}")

            if best_metric is None or current > best_metric + ES_MIN_DELTA * 100.0:
                best_metric = current
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= ES_PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        ckpt_dir = os.path.join(SAVE_CKPT_DIR, f"{DOMAIN.lower()}_seed{seed}")
        pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))

    infer_and_eval_presence(seed, ds_test, model, test_loader, gold_merged)

def infer_and_eval_presence(seed: int, ds_test: DomBertDataset, model, loader, gold_merged):
    ensure_dir(OUT_DIR)
    device = next(model.parameters()).device

    pred_wide_csv = os.path.join(OUT_DIR, f"{DOMAIN.lower()}_seed{seed}_pred_wide.csv")
    met_csv       = os.path.join(OUT_DIR, f"{DOMAIN.lower()}_seed{seed}_metrics_presence.csv")

    perfile = {}

    with open(pred_wide_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["file","name","duration","stops","price","departure_time","arrival_time","airline"])

        model.eval()
        with torch.no_grad():
            for input_ids, attn, labels, meta in tqdm(loader, desc=f"Seed {seed} Infer"):
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                logits, _ = model(input_ids=input_ids, attention_mask=attn, labels=None)
                dec = decode_bio_to_sets(meta[0], logits[0].cpu())

                if USE_REGEX_BACKOFF:
                    raw = read_file(meta[0]["file"]) if os.path.isfile(meta[0]["file"]) else ""
                    plain = strip_html_to_text(raw)
                    rb = regex_backoff_candidates(plain, raw_html=raw)
                    for lab in PRESENCE_LABELS:
                        if not dec.get(lab) and rb.get(lab):
                            dec[lab] = rb[lab]

                perfile[meta[0]["file"]] = dec

                def _one(lab: str) -> str:
                    vals = dec.get(lab, [])
                    if not vals:
                        return ""
                    v = vals[0]
                    if lab == "PRICE":
                        return normalize_price(v) or v
                    if lab == "DURATION":
                        return normalize_duration(v) or v
                    if lab == "STOPS":
                        return normalize_stops(v) or v
                    if lab in ("DEPARTURE_TIME","ARRIVAL_TIME"):
                        return normalize_time_value(v) or v
                    if lab == "AIRLINE":
                        return normalize_airline(v) or v
                    return v

                w.writerow([
                    meta[0]["file"],
                    _one("NAME"),
                    _one("DURATION"),
                    _one("STOPS"),
                    _one("PRICE"),
                    _one("DEPARTURE_TIME"),
                    _one("ARRIVAL_TIME"),
                    _one("AIRLINE"),
                ])

    print(f"[{DOMAIN}] Seed {seed} wide predictions → {pred_wide_csv}")

    totals = {"files":0, "all7_sum":0}
    sums = {lab:0 for lab in PRESENCE_LABELS}

    with open(met_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["seed","file"] + [f"{lab.lower()}_presence" for lab in PRESENCE_LABELS] + ["all7_presence"])

        for ex in ds_test.items:
            base = os.path.basename(ex.file)
            gold = gold_merged.get(base)
            if gold is None:
                continue
            preds = perfile.get(ex.file, {})

            g = {lab: bool(gold.get(lab)) for lab in PRESENCE_LABELS}
            p = {lab: bool(preds.get(lab)) for lab in PRESENCE_LABELS}

            row_scores = []
            ok_all = True
            for lab in PRESENCE_LABELS:
                sc = 100 if (g[lab] == p[lab]) else 0
                row_scores.append(sc)
                sums[lab] += sc
                if sc != 100:
                    ok_all = False

            all7 = 100 if ok_all else 0
            totals["files"] += 1
            totals["all7_sum"] += all7

            w.writerow([seed, ex.file] + row_scores + [all7])

        if totals["files"] > 0:
            summary = [f"{(sums[lab]/totals['files']):.2f}" for lab in PRESENCE_LABELS]
            w.writerow([seed, "__SUMMARY__"] + summary + [f"{(totals['all7_sum']/totals['files']):.2f}"])

    print(f"[{DOMAIN}] Seed {seed} presence metrics → {met_csv}")

def aggregate_presence_across_seeds(out_dir: str, seeds: List[int]):
    summaries = []
    for sd in seeds:
        p = os.path.join(out_dir, f"{DOMAIN.lower()}_seed{sd}_metrics_presence.csv")
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as fr:
            reader = csv.DictReader(fr)
            for row in reader:
                if row["file"] == "__SUMMARY__":
                    rec = {"seed": sd}
                    for lab in PRESENCE_LABELS:
                        rec[lab.lower()] = float(row[f"{lab.lower()}_presence"])
                    rec["all7"] = float(row["all7_presence"])
                    summaries.append(rec)

    if not summaries:
        print("No presence summaries to aggregate.")
        return

    sum_csv = os.path.join(out_dir, f"{DOMAIN.lower()}_presence_summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["seed"] + [f"mean_{lab.lower()}_presence" for lab in PRESENCE_LABELS] + ["mean_all7_presence"])
        for r in sorted(summaries, key=lambda x: x["seed"]):
            w.writerow([r["seed"]] + [f"{r[lab.lower()]:.2f}" for lab in PRESENCE_LABELS] + [f"{r['all7']:.2f}"])

    agg_csv = os.path.join(out_dir, f"{DOMAIN.lower()}_presence_aggregate.csv")
    with open(agg_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        cols = [lab.lower() for lab in PRESENCE_LABELS] + ["all7"]
        w.writerow(["domain","seeds"] +
                   [f"mean_{c}" for c in cols] +
                   [f"std_{c}" for c in cols] +
                   ["note"])

        means = {c: statistics.mean([r[c] for r in summaries]) for c in cols}
        stds  = {c: (statistics.pstdev([r[c] for r in summaries]) if len(summaries) > 1 else 0.0) for c in cols}

        w.writerow([
            DOMAIN,
            ",".join(str(r["seed"]) for r in sorted(summaries, key=lambda x: x["seed"])),
            *[f"{means[c]:.2f}" for c in cols],
            *[f"{stds[c]:.2f}" for c in cols],
            "Presence on test files with gold; NAME/DURATION/STOPS/PRICE/DEPARTURE_TIME/ARRIVAL_TIME/AIRLINE + ALL7"
        ])

    print(f"✓ Wrote per-seed presence summary: {sum_csv}")
    print(f"✓ Wrote presence aggregate:       {agg_csv}")

# =========================
# Main
# =========================
def main():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    files = list_html_files(HTML_DIR)
    if not files:
        raise RuntimeError(f"No HTML in {HTML_DIR}")
    for sd in SEEDS:
        train_one_seed(sd)
    aggregate_presence_across_seeds(OUT_DIR, SEEDS)

if __name__ == "__main__":
    main()
