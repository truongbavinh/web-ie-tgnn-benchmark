# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Optional, Any
import re, unicodedata
from bs4 import BeautifulSoup

def _text(el) -> str:
    if not el: return ""
    return " ".join(el.get_text(" ", strip=True).split())

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

def _pick_first(soup, selectors: List[str]):
    for css in selectors:
        try:
            if ":contains(" in css:
                m = re.match(r'^(.*?)\:contains\("(.+?)"\)$', css)
                if not m: continue
                base, kw = m.group(1), m.group(2)
                for el in soup.select(base):
                    if kw.lower() in _text(el).lower():
                        return el
            else:
                el = soup.select_one(css)
                if el: return el
        except Exception:
            continue
    return None
def extract_fields(html: str, url: Optional[str]=None):
    soup = BeautifulSoup(html, "html.parser")
    host = ""
    if url:
        import urllib.parse as up
        try: host = up.urlparse(url).netloc.lower()
        except: host = ""
    sel = {
      "kayak.com": {
        "name": ['[data-resultid]', '.itinerary-details'],
        "duration": ['[class*="duration"]'],
        "stops": ['[class*="stops"]'],
        "price": ['[class*="price"]'],
        "departure_time": ['[class*="depart-time"]','.depart-time'],
        "arrival_time": ['[class*="arrival-time"]','.arrival-time'],
        "airline": ['[class*="airline"]'],
      },
      "tripadvisor.com": {
        "name": ['h1','.route-title'],
        "duration": ['[class*="duration"]'],
        "stops": ['[class*="stops"]'],
        "price": ['[class*="price"]'],
        "departure_time": ['[class*="depart"]'],
        "arrival_time": ['[class*="arrive"]'],
        "airline": ['[class*="airline"]'],
      }
    }
    fallback = {
      "name": ['h1','h2'],
      "duration": ['[class*="duration"]'],
      "stops": ['[class*="stop"]'],
      "price": ['[class*="price"]'],
      "departure_time": ['[class*="depart"]'],
      "arrival_time": ['[class*="arriv"]'],
      "airline": ['[class*="airline"]'],
    }
    s = sel.get(host, fallback)
    def g(k): return _text(_pick_first(soup, s.get(k, []))) or None
    rec = {k:g(k) for k in ["name","duration","stops","price","departure_time","arrival_time","airline"]}
    # stops -> int if possible
    import re
    if isinstance(rec["stops"], str):
        m = re.search(r'[0-9]+', rec["stops"])
        rec["stops"] = int(m.group(0)) if m else 0
    return rec
